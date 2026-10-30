"""Read-only probes for the 2026-10-29 review; temporary corpus, no live LLM.

Run from the repository root:
    python docs/research/single-model-rag-review-2026-10-29.probes.py

The depth probe supplies a fixed lexical ranking to the real discovery/version
pipeline. It tests candidate wiring, not retrieval quality or model behavior.
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.agentic.discovery import AgenticDiscovery, select_chunks_for_focus
from app.agentic.evidence_view import EvidenceDriller
from app.agentic.service import AgenticReasoningService
from app.generation.citation_check import check_citations
from app.retrieval.context_builder import build_contexts
from app.retrieval.hits import SearchHit
from app.schemas.chunk import Chunk
from app.schemas.document import Document
from app.wiki.compiler import WikiCompiler
from app.wiki.search import WikiSearcher


def chunk(doc_id, index, text):
    return Chunk(chunk_id=f"{doc_id}:{index}", doc_id=doc_id,
                 chunk_index=index, text=text, source_path=f"{doc_id}.md",
                 title=f"Section {index}", metadata={"kb_id": "default"})


def article_filters(root):
    compiler = WikiCompiler(root / "wiki")
    chunks = [chunk("doc-policy", i, "General administrative background. " * 20)
              for i in range(9)]
    chunks.append(chunk("doc-policy", 9, "Quasarneedle requests must be filed within 17 days."))
    doc = Document(doc_id="doc-policy", source_path="policy.md", title="Operations",
                   content="\n\n".join(c.text for c in chunks),
                   metadata={"kb_id": "default", "doc_type": "policy", "source_key": "policy",
                             "version": "v1", "effective_date": "2025-01-01"})
    compiler.upsert_document(doc, chunks)
    compiler.write_article_pages(doc, [("Clause 10", chunks[-1].text)])
    searcher = WikiSearcher(root / "wiki")
    def ids(filters):
        return [(h.chunk.doc_id, h.chunk.metadata["wiki_kind"])
                for h in searcher.search("Quasarneedle", 10, metadata_filters=filters,
                                         granularity="document")]
    return {"without_filter": ids(None),
            "matching_doc_type_filter": ids({"doc_types": ["policy"]}),
            "matching_date_filter": ids({"effective_date_to": "2026-01-01"})}


async def depth_probe():
    hits = []
    for i in range(1, 101):
        c = chunk(f"doc-{i:03}", 0, "policy text")
        c.metadata["wiki_kind"] = "source"
        hits.append(SearchHit(chunk=c, score=101 - i))
    fetches = []
    d = AgenticDiscovery.__new__(AgenticDiscovery)
    async def search(query, *, fetch_k, **kwargs):
        fetches.append(fetch_k)
        return hits[:fetch_k]
    d._search = search
    d.discovery_top_k = 12
    d.scope_preference_enabled = False
    d.query_expansion_enabled = False
    d.hyde_enabled = False
    d.llm_rerank_enabled = False
    d.candidate_filter_enabled = False
    d.include_stale_default = False
    r1 = await d.discovery_candidates("policy", depth=1)
    r2 = await d.discovery_candidates("policy rewritten", depth=2)
    return {"search_fetch_k": fetches,
            "round1_candidate_ids": [h.chunk.doc_id for h in r1.candidates],
            "round2_candidate_ids": [h.chunk.doc_id for h in r2.candidates],
            "new_round2_candidates": len({h.chunk.doc_id for h in r2.candidates}
                                          - {h.chunk.doc_id for h in r1.candidates})}


def prefix_probe():
    c = chunk("long", 0, "Ordinary background. " * 330 + " Refund deadline is 17 days.")
    contexts = build_contexts([SearchHit(chunk=c, score=1)], 1, max_text_chars=6000)
    excerpts = EvidenceDriller().excerpt(contexts, watch_terms=["Refund deadline"])
    return {"source_length": len(c.text), "context_length": len(contexts[0]["text"]),
            "source_contains_answer": "17 days" in c.text,
            "context_contains_answer": "17 days" in contexts[0]["text"],
            "excerpt_exhausted": excerpts[0].exhausted}


def selection_probe():
    chunks = [chunk("long", i, f"Paragraph {i}.").model_dump() for i in range(25)]
    selected = select_chunks_for_focus(chunks, set(), doc_id="long", max_chunks=24)
    contexts = [{"chunk_id": f"a:{i}", "source": "a.md", "text": "Background."}
                for i in range(24)]
    contexts.append({"chunk_id": "b:0", "source": "b.md", "text": "The deadline is 17 days."})
    visible = EvidenceDriller().excerpt(contexts, watch_terms=["deadline"])
    return {"raw_chunks": len(chunks), "selected_chunks": len(selected),
            "last_chunk_selected": any(c["chunk_id"] == "long:24" for c in selected),
            "visible_sources": sorted({e.source for e in visible}),
            "answer_source_visible": any(e.source == "b.md" for e in visible)}


def view_probe():
    text = "General policy information is provided here. " * 55
    text += "\n\nEscrow reimbursement deadline is 17 days.\n\n"
    text += "Unrelated maintenance instructions are listed here. " * 55
    contexts = [{"chunk_id": "a", "citation_label": "C1", "source": "a.md", "text": text}]
    driller = EvidenceDriller(total_chars=1200, max_per_context=1200, min_per_context=100)
    audited = driller.excerpt(contexts, watch_terms=["policy", "escrow"])[0]
    service = AgenticReasoningService.__new__(AgenticReasoningService)
    service.evidence_driller = driller
    synthesis = service._synthesis_contexts("policy", contexts)[0]
    return {"same_view": audited.render() == synthesis["text"],
            "auditor_has_deadline": "17 days" in audited.render(),
            "synthesis_has_deadline": "17 days" in synthesis["text"],
            "auditor_offsets": [p.offset for p in audited.passages],
            "synthesis_offsets": [p["offset"] for p in synthesis["evidence_excerpt"]["passages"]]}


async def graph_probe():
    class Expander:
        calls = 0
        def expand(self, *args, **kwargs):
            self.calls += 1
            return [{"text": "linked evidence"}]
    from app.schemas.chat import ChatRequest
    s = AgenticReasoningService.__new__(AgenticReasoningService)
    s.graph_expansion_mode = "auto"
    s.graph_expander = Expander()
    normal = await s._expand_graph_contexts(ChatRequest(query="linked policy"),
        [{"node_id": "a"}], {"retrieval_params": {"engine": "agentic_wiki"}})
    return {"normal_trace_expands": bool(normal), "expander_calls": s.graph_expander.calls}


def stale_probe():
    old = chunk("old-version", 0, "Legacy procedure.")
    old.metadata.update(source_key="policy", version="v1", effective_date="2024-01-01",
                        is_latest_version=False, superseded_by="new-version")
    d = AgenticDiscovery.__new__(AgenticDiscovery)
    d.include_stale_default = False
    d.discovery_top_k = 12
    selected, _ = d._select_versions([SearchHit(chunk=old, score=1)])
    return {"include_stale": d.include_stale_default,
            "input_is_latest": old.metadata["is_latest_version"],
            "selected_doc_ids": [h.chunk.doc_id for h in selected]}


def integrity_probe():
    source = [{"citation_label": "C1", "text": "The refund deadline is 17 days. The claim limit is 99 dollars."}]
    wrong = check_citations("The refund deadline is 99 days [C1].", source)
    return {"wrong_answer": "The refund deadline is 99 days [C1].",
            "integrity_score": wrong.integrity_score,
            "citation_precision": wrong.citation_precision,
            "issue_count": len(wrong.issues)}


async def main():
    with tempfile.TemporaryDirectory(prefix="nano-rag-review-") as temp:
        report = {"article_filters": article_filters(Path(temp)),
                  "depth": await depth_probe(), "prefix": prefix_probe(),
                  "chunk_selection": selection_probe(), "audit_synthesis_views": view_probe(),
                  "graph_auto": await graph_probe(), "stale_version": stale_probe(),
                  "citation_semantics": integrity_probe()}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
