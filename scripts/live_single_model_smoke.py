"""Live query-chain smoke on temporary source artifacts; run inside Docker.

Uses only env-configured model credentials. No running service data is changed.
This verifies the query chain, not document parser quality or benchmark accuracy.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agentic.service import AgenticReasoningService
from app.core.config import load_config
from app.core.tracing import TraceStore, TracingManager
from app.core.usage import collect_model_usage
from app.generation.answer_formatter import AnswerFormatter
from app.generation.prompt_builder import PromptBuilder
from app.model_client.generation import GenerationClient
from app.schemas.chat import ChatRequest
from app.schemas.chunk import Chunk
from app.schemas.document import Document
from app.wiki.compiler import WikiCompiler
from app.wiki.search import WikiSearcher
from app.wiki.knowledge import compile_knowledge


class SmokeClient(GenerationClient):
    """Diagnostics contain only outputs from this script's synthetic corpus."""
    def __init__(self, config):
        super().__init__(config)
        self.outputs = {}

    async def generate(self, messages, **kwargs):
        result = await super().generate(messages, **kwargs)
        if kwargs.get("usage_stage") in {"evidence_audit", "claim_grounding", "answer_synthesis"}:
            self.outputs[kwargs["usage_stage"]] = result.get("content")
        return result


async def main():
    for name in ("GENERATION_MODEL_ALIAS", "GENERATION_API_BASE_URL", "GENERATION_API_KEY"):
        if not os.getenv(name) and os.getenv("COMPOSE_" + name):
            os.environ[name] = os.environ["COMPOSE_" + name]
    if not os.getenv("GENERATION_MODEL_ALIAS"):
        raise RuntimeError("GENERATION_MODEL_ALIAS must be supplied through env")
    for flag in ("RAG_QUERY_PLAN_ENABLED", "RAG_NODE_NAVIGATION_ENABLED", "RAG_CLAIM_VERIFICATION_ENABLED"):
        os.environ[flag] = "true"
    with tempfile.TemporaryDirectory(prefix="nano-rag-live-") as temp:
        root = Path(temp)
        os.environ["PARSED_OUTPUT_DIR"] = str(root/"parsed")
        (root/"parsed").mkdir()
        config = load_config()
        client = SmokeClient(config)
        source = "Administrative background. " * 350 + "\n\nThe refund deadline is 17 days. The claim limit is 99 dollars."
        doc = Document(doc_id="live-policy", source_path="live-policy.md", title="Refund policy",
            content=source, metadata={"kb_id":"live-smoke", "doc_type":"policy", "source_key":"refund-policy",
                                      "version":"1", "effective_date":"2026-01-01"})
        chunk = Chunk(chunk_id="live-policy:node:0", doc_id=doc.doc_id, chunk_index=0,
            source_path=doc.source_path, title=doc.title, text=source,
            metadata={**doc.metadata, "hierarchy_path":[doc.title]})
        compilation = None
        compilation_usage = None
        if "--compile" in sys.argv:
            with collect_model_usage() as ledger:
                compilation = await compile_knowledge(doc, [chunk], client)
            compilation_usage = ledger.total()
            compilation = {key: value for key, value in compilation.items() if key != "model"}
        (root/"parsed"/f"{doc.doc_id}.json").write_text(json.dumps({"document":doc.model_dump(),
            "chunks":[chunk.model_dump()],"structured_document":None}))
        compiler = WikiCompiler(root/"wiki")
        compiler.upsert_document(doc, [chunk])
        compiler.write_article_pages(doc, [("Refund deadline", source[source.index("The refund"):])], chunks=[chunk])
        searcher = WikiSearcher(root/"wiki")
        traces = TraceStore()
        service = AgenticReasoningService(config, client, PromptBuilder(config.prompts, config.settings),
            AnswerFormatter(), traces, TracingManager("nano-rag-live-smoke", ""), wiki_searcher=searcher)
        try:
            response = await service.run(ChatRequest(query="What is the refund deadline?", kb_id="live-smoke"))
            trace = traces.get(response.trace_id)
            if "17" not in response.answer or not response.grounded_claims or not trace.answerable:
                print(json.dumps({"status":"failed", "answer":response.answer,
                    "answerable":trace.answerable, "retrieval":trace.retrieval_params}, ensure_ascii=False), flush=True)
                raise RuntimeError("live model did not produce a grounded answer from the tail source passage")
            view = trace.retrieval_params["agent"]["evidence_view"]
            if view["visible_chars"] > service.evidence_driller.max_per_context:
                raise RuntimeError("evidence snapshot exceeded its configured source budget")
            for citation in response.citations:
                if source[citation.span_start:citation.span_end] != citation.span_text:
                    raise RuntimeError("response citation did not map back to the original source")
            print(json.dumps({"status":"passed", "model_from_env":True,
                "model_alias_count":len({c["model"] for c in trace.model_calls}),
                "answer":response.answer, "grounded_claims":len(response.grounded_claims),
                "model_calls":len(trace.model_calls), "usage":trace.generation_usage,
                "model_stages":[c["stage"] for c in trace.model_calls],
                "lexical_backend":os.getenv("RAG_LEXICAL_BACKEND", "memory"),
                "knowledge_compilation":compilation,
                "knowledge_compilation_usage":compilation_usage,
                "latency_seconds":trace.latency_seconds,
                "citation_integrity":trace.citation_integrity,
                "source_answer_offset":source.index("The refund"),
                "citation_source_start":response.citations[0].span_start,
                "evidence_view":trace.retrieval_params["agent"]["evidence_view"]}, ensure_ascii=False))
        except Exception as exc:
            print(json.dumps({"status":"failed", "error_type":type(exc).__name__,
                "error":str(exc), "synthetic_model_outputs":client.outputs}, ensure_ascii=False), flush=True)
            raise
        finally:
            await client.close()
            if hasattr(searcher.index, "close"):
                searcher.index.close()


if __name__ == "__main__":
    asyncio.run(main())
