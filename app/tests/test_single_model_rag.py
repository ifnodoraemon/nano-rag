"""Cross-stage regressions and the default single-model path, using real indexes."""
from __future__ import annotations

import json
from contextlib import contextmanager
from time import perf_counter
from types import SimpleNamespace

import pytest

from app.agentic.discovery import AgenticDiscovery
from app.agentic.evidence_set import EvidenceSet
from app.agentic.evidence_view import EvidenceDriller
from app.agentic.query_plan import QueryPlan
from app.agentic.read_tools import ReadAction, ReadTools
from app.agentic.service import AgenticReasoningService
from app.core.exceptions import ConfigurationError, ModelOutputError, RetrievalError
from app.core.tracing import TraceStore
from app.core.usage import collect_model_usage, model_stage, record_model_call, reserve_model_call
from app.generation.answer_formatter import AnswerFormatter
from app.generation.citation_check import check_citations
from app.generation.grounding import verify_answer
from app.generation.prompt_builder import PromptBuilder
from app.retrieval.artifacts import ArtifactReader
from app.retrieval.bm25 import BM25Config, BM25Index
from app.retrieval.context_builder import build_contexts
from app.retrieval.hits import SearchHit
from app.retrieval.sqlite_index import SQLiteBM25Index
from app.schemas.chat import ChatRequest
from app.schemas.chunk import Chunk
from app.tests.test_agentic_discovery import _doc, _write_artifact
from app.wiki.compiler import WikiCompiler
from app.wiki.knowledge import compile_knowledge
from app.wiki.search import WikiSearcher
from app.core.config import AppConfig


def corpus(tmp_path, *, count=30, kb="default"):
    parsed = tmp_path / "parsed"
    parsed.mkdir(exist_ok=True)
    chunks = [Chunk(chunk_id=f"policy:{i}", doc_id="policy", chunk_index=i,
                    title=f"Section {i}", source_path="policy.md",
                    text="Administrative background." if i != count-1 else "Quasarneedle refund deadline is 17 days.",
                    metadata={"kb_id": kb, "hierarchy_path": [f"Section {i}"]}) for i in range(count)]
    doc = _doc("policy", "policy-family", "v1", "2025-01-01", "\n\n".join(c.text for c in chunks))
    doc.metadata["kb_id"] = kb
    compiler = WikiCompiler(tmp_path / "wiki")
    compiler.upsert_document(doc, chunks)
    compiler.write_article_pages(doc, [("Section 29", chunks[-1].text)], chunks=chunks)
    _write_artifact(parsed, "policy", doc, chunks)
    return WikiSearcher(tmp_path / "wiki"), doc, chunks


def test_filters_keep_article_and_parent_version(tmp_path):
    searcher, _, _ = corpus(tmp_path)
    for filters in ({"doc_types": ["policy"]}, {"effective_date_to": "2026-01-01"}, {"version": "1"}):
        hits = searcher.search("Quasarneedle", 10, metadata_filters=filters)
        assert hits and hits[0].chunk.metadata["wiki_kind"] == "article"
        assert hits[0].chunk.metadata["matched_node_ids"] == ["policy:29"]


def test_document_discovery_retains_multiple_seeds_and_widens_past_article_siblings(tmp_path):
    searcher, doc, chunks = corpus(tmp_path, count=30)
    for chunk in chunks:
        chunk.text = f"Quasarneedle unique section {chunk.chunk_index}."
    doc.content = "\n\n".join(c.text for c in chunks)
    compiler = WikiCompiler(tmp_path/"wiki")
    compiler.upsert_document(doc, chunks)
    compiler.write_article_pages(doc, [(f"Section {i}", c.text) for i, c in enumerate(chunks)], chunks=chunks)
    small = _doc("short", "short-family", "v1", "2025-01-01", "Quasarneedle " + "filler "*200)
    compiler.upsert_document(small, [Chunk(chunk_id="short:0", doc_id="short", chunk_index=0,
        source_path="short.md", text=small.content)])
    searcher.sync()
    hits = searcher.search("Quasarneedle", 2, granularity="document", metadata_filters={"version":"1"})
    assert {h.chunk.doc_id for h in hits} == {"policy", "short"}
    policy = next(h for h in hits if h.chunk.doc_id == "policy")
    assert len(policy.chunk.metadata["matched_node_ids"]) > 1
    assert len(policy.chunk.metadata["matched_pages"]) > 8


def test_reading_targets_late_node_and_preserves_full_text(tmp_path):
    searcher, _, chunks = corpus(tmp_path)
    d = AgenticDiscovery(config=SimpleNamespace(parsed_dir=tmp_path/"parsed", settings={}),
        wiki_searcher=searcher, generation_client=None, trace_store=None, tracing_manager=None)
    metadata = searcher.search("Quasarneedle", 1)[0].chunk.metadata
    contexts = d._contexts_for_document("policy", set(), {**metadata, "read_query": "refund deadline"})
    assert contexts[0]["chunk_id"] == chunks[-1].chunk_id
    assert contexts[0]["document_read_complete"] is False
    long = chunks[-1].model_copy(update={"text": "Background. " * 700 + " Tail answer is 17 days."})
    context = build_contexts([SearchHit(chunk=long, score=1)], 1, truncate=False)[0]
    assert "17 days" in EvidenceDriller().excerpt([context], watch_terms=["Tail answer"])[0].render()
    assert "17 days" in EvidenceDriller().excerpt([context], watch_terms=["What is the tail answer?"])[0].render()
    context["text"] = "Administrative background. " * 350 + "The refund deadline is 17 days."
    assert "17 days" in EvidenceDriller().excerpt([context], watch_terms=["What is the refund deadline?"])[0].render()
    bounded = EvidenceDriller(total_chars=3000, max_per_context=3000).excerpt(
        [context], watch_terms=["Administrative background", "refund deadline"])[0]
    assert bounded.visible_chars <= 3000
    assert "17 days" in bounded.render()


def test_snapshot_is_immutable_and_balances_documents():
    contexts = [{"doc_id": "a", "chunk_id": f"a:{i}", "citation_label": f"C{i+1}",
                 "text": "Background.", "source": "a"} for i in range(24)]
    contexts.append({"doc_id": "b", "chunk_id": "b:0", "citation_label": "C25",
                     "text": "Refund deadline is 17 days.", "source": "b"})
    snapshot = EvidenceSet.build(contexts, EvidenceDriller(), ["refund deadline"])
    assert any(c["doc_id"] == "b" for c in snapshot.contexts())
    rendered = snapshot.contexts()
    rendered[0]["text"] = "mutated"
    contexts[0]["text"] = "also mutated"
    assert all("mutated" not in c["text"] for c in snapshot.contexts())
    assert not snapshot.complete


@pytest.mark.asyncio
async def test_response_citation_uses_durable_source_offset():
    text = "Administrative background. "*350 + "The refund deadline is 17 days."
    context = {"doc_id":"d", "chunk_id":"n", "node_id":"n", "citation_label":"C1",
               "text":text, "source":"policy.md"}
    snapshot = EvidenceSet.build([context], EvidenceDriller(), ["refund deadline"])
    claims = await verify_answer("The refund deadline is 17 days [C1].", snapshot.contexts(), ScriptedModel())
    response = AnswerFormatter().format("The refund deadline is 17 days [C1].", snapshot.contexts(),
                                         None, grounded_claims=claims)
    citation = response.citations[0]
    assert citation.span_start == text.index("refund deadline is 17 days")
    assert text[citation.span_start:citation.span_end] == citation.span_text


def test_truncated_input_cannot_be_reported_complete():
    chunk = Chunk(chunk_id="a", doc_id="d", chunk_index=0, source_path="d", text="x"*7000)
    contexts = build_contexts([SearchHit(chunk=chunk, score=1)], 1)
    assert not EvidenceSet.build(contexts, EvidenceDriller(), ["x"]).complete


def test_wrong_numeric_unit_binding_is_not_perfect():
    report = check_citations("The refund deadline is 99 days [C1].", [{"citation_label": "C1",
        "text": "The refund deadline is 17 days. The claim limit is 99 dollars."}])
    assert report.integrity_score == 0
    assert any(i.kind == "numeric_binding" for i in report.issues)


@pytest.mark.parametrize("backend", [BM25Index, SQLiteBM25Index])
def test_fielded_lexical_index_scope_replace_and_delete(backend):
    index = backend(BM25Config())
    try:
        index.add_document("a", "ordinary text", fields={"questions": "Quasarneedle refund"})
        index.add_document("b", "Quasarneedle")
        assert index.search("Quasarneedle", 2, allowed_doc_ids={"a"})[0][0] == "a"
        index.add_document("a", "replacement")
        assert index.search("Quasarneedle", 2, allowed_doc_ids={"a"}) == []
        index.remove_document("b")
        assert index.search("Quasarneedle", 2) == []
        index.clear()
        assert index.document_count == 0
    finally:
        if hasattr(index, "close"):
            index.close()


def test_artifact_tools_cannot_escape_scope(tmp_path):
    searcher, _, _ = corpus(tmp_path, kb="private")
    reader = ArtifactReader(tmp_path/"parsed")
    with pytest.raises(RetrievalError):
        reader.load("../outside", "private")
    with pytest.raises(RetrievalError):
        reader.read_nodes("policy", ["policy:29"], "default")
    with pytest.raises(RetrievalError):
        reader.read_nodes("policy", ["fabricated"], "private")
    tools = ReadTools(tmp_path/"parsed", searcher)
    with pytest.raises(RetrievalError):
        tools.execute(ReadAction(kind="read_nodes", doc_id="policy", node_ids=["policy:29"]),
                      "private", {"doc_types": ["unrelated"]})


def test_query_plan_checks_temporal_scope_and_dependencies():
    with pytest.raises(ValueError):
        QueryPlan(query="q", temporal="as_of")
    with pytest.raises(ValueError):
        QueryPlan(query="q", subgoals=[{"id":"b", "question":"q", "depends_on":["a"]}])
    assert QueryPlan(query="q", temporal="as_of", as_of="2020-01-01").retrieval()["metadata_filters"]["effective_date_to"] == "2020-01-01"


class ScriptedModel:
    alias = "one-model"
    def __init__(self):
        self.calls = []
    async def generate(self, messages, **kwargs):
        name = kwargs.get("response_format", {}).get("json_schema", {}).get("name")
        self.calls.append(name)
        reserve_model_call()
        with model_stage(kwargs.get("usage_stage", "test")):
            record_model_call(self.alias, {"total_tokens": 10}, perf_counter())
        if name == "query_plan":
            payload = QueryPlan(query="Quasarneedle refund deadline").model_dump()
        elif name == "reading_plan":
            data = json.loads(messages[-1]["content"].split("Input JSON: ")[1])
            doc = data["candidates"][0]
            payload = {"selected_docs": [{"doc_id": doc["doc_id"], "focus_sections": [],
                         "reason": "exact evidence", "node_ids": ["policy:29"]}]}
        elif name == "evidence_verification":
            payload = {"sufficient": True, "coverage_ratio": 1, "missing_terms": [],
                       "follow_up_queries": [], "reason": "source found", "actions": [],
                       "facts": [], "completed_subgoals": []}
        elif name == "answer_structure":
            payload = {"is_answerable": True, "missing_entities": [],
                "extracted_answer": "The refund deadline is 17 days [C1].",
                "supporting_claims": ["[factual] The refund deadline is 17 days [C1]."]}
        elif name == "claim_grounding":
            payload = {"verdict": "supported", "claims": [{"sentence_index": 0, "supported": True,
                "label": "C1", "quote": "refund deadline is 17 days", "entity": "refund",
                "attribute": "deadline", "value": "17", "unit": "days", "conditions": []}]}
        else:
            raise AssertionError(name)
        return {"content": json.dumps(payload), "usage": {"total_tokens": 10}, "finish_reason": "stop"}


@pytest.mark.asyncio
async def test_default_path_same_snapshot_full_usage_and_cache(tmp_path, monkeypatch):
    for flag in ("RAG_QUERY_PLAN_ENABLED", "RAG_NODE_NAVIGATION_ENABLED", "RAG_CLAIM_VERIFICATION_ENABLED"):
        monkeypatch.setenv(flag, "true")
    searcher, _, _ = corpus(tmp_path)
    config = SimpleNamespace(parsed_dir=tmp_path/"parsed", settings={
        "agent": {"max_retrieval_loops": 2}, "prompt": {"version": "test"}})
    @contextmanager
    def span(*args, **kwargs):
        yield
    client = ScriptedModel()
    traces = TraceStore()
    service = AgenticReasoningService(config, client, PromptBuilder({}), AnswerFormatter(), traces,
                                      SimpleNamespace(span=span), wiki_searcher=searcher)
    response = await service.run(ChatRequest(query="Quasarneedle refund deadline"))
    assert response.usage["model_calls"] == 5
    assert response.usage["total_tokens"] == 50
    assert response.grounded_claims[0]["node_id"] == "policy:29"
    trace = traces.get(response.trace_id)
    assert trace.latency_seconds >= trace.step_latencies["agentic_discovery_seconds"]
    assert trace.retrieval_params["agent"]["rounds"][0]["evidence_view"]["evidence_ids"]
    assert len(trace.model_calls) == 5
    again = await service.run(ChatRequest(query="Quasarneedle refund deadline"))
    assert again.usage["model_calls"] == 4


@pytest.mark.asyncio
async def test_grounding_rejects_model_invented_quote():
    with pytest.raises(ModelOutputError, match="absent"):
        await verify_answer("The refund deadline is 17 days [C1].",
            [{"citation_label": "C1", "text": "Unrelated text"}], ScriptedModel())


@pytest.mark.asyncio
async def test_grounding_canonicalizes_labels_and_covers_uncited_sentences():
    class BracketModel(ScriptedModel):
        async def generate(self, messages, **kwargs):
            result = await super().generate(messages, **kwargs)
            payload = json.loads(result["content"])
            payload["claims"][0]["label"] = "[C1]"
            return {**result, "content": json.dumps(payload)}
    contexts = [{"citation_label": "C1", "text": "The refund deadline is 17 days."}]
    claims = await verify_answer("The refund deadline is 17 days [C1].", contexts, BracketModel())
    assert claims[0]["label"] == "C1"
    service = AgenticReasoningService.__new__(AgenticReasoningService)
    facts = service._validated_facts([{"text":"refund deadline", "label":"[C1]",
                                      "quote":"refund deadline is 17 days"}], contexts)
    assert facts[0]["label"] == "C1"
    with pytest.raises(ModelOutputError, match="not anchored"):
        service._validated_facts([{"text":"fee", "label":"C1", "quote":"invented fee"}], contexts)
    with pytest.raises(ModelOutputError, match="every answer sentence"):
        await verify_answer("The refund deadline is 17 days [C1]. It is always waived.",
                            contexts, BracketModel())


@pytest.mark.asyncio
async def test_media_bytes_are_frozen_across_audit_synthesis_and_grounding(tmp_path, monkeypatch):
    import base64
    for flag in ("RAG_QUERY_PLAN_ENABLED", "RAG_NODE_NAVIGATION_ENABLED", "RAG_CLAIM_VERIFICATION_ENABLED"):
        monkeypatch.setenv(flag, "true")
    searcher, doc, chunks = corpus(tmp_path)
    image = tmp_path / "policy.png"
    image.write_bytes(b"source-image")
    chunks[-1] = chunks[-1].model_copy(update={"modality": "image", "media_uri": str(image),
                                              "mime_type": "image/png"})
    _write_artifact(tmp_path/"parsed", "policy", doc, chunks)
    expected = "data:image/png;base64," + base64.b64encode(b"source-image").decode()
    stages = []
    class MediaModel(ScriptedModel):
        async def generate(self, messages, **kwargs):
            name = kwargs.get("response_format", {}).get("json_schema", {}).get("name")
            if name in {"evidence_verification", "answer_structure", "claim_grounding"}:
                assert messages[-1]["content"][-1]["image_url"]["url"] == expected
                if name == "answer_structure":
                    assert "refund deadline is 17 days" in messages[-1]["content"][0]["text"]
                stages.append(name)
                image.write_bytes(b"changed-after-audit")
            return await super().generate(messages, **kwargs)
    @contextmanager
    def span(*args, **kwargs):
        yield
    config = SimpleNamespace(parsed_dir=tmp_path/"parsed", settings={"prompt":{"version":"test"}})
    traces = TraceStore()
    service = AgenticReasoningService(config, MediaModel(), PromptBuilder({}), AnswerFormatter(),
                                      traces, SimpleNamespace(span=span), wiki_searcher=searcher)
    response = await service.run(ChatRequest(query="Quasarneedle refund deadline"))
    assert stages == ["evidence_verification", "answer_structure", "claim_grounding"]
    view = traces.get(response.trace_id).retrieval_params["agent"]["evidence_view"]
    assert view["media_parts"] == 1 and len(view["media_sha256"]) == 1


def test_call_budget_is_shared_and_fails_loudly():
    with collect_model_usage(max_calls=1):
        reserve_model_call()
        with pytest.raises(ModelOutputError, match="budget"):
            reserve_model_call()


@pytest.mark.asyncio
async def test_corrective_read_uses_new_node_and_keeps_round_evidence(tmp_path, monkeypatch):
    for flag in ("RAG_QUERY_PLAN_ENABLED", "RAG_NODE_NAVIGATION_ENABLED", "RAG_CLAIM_VERIFICATION_ENABLED"):
        monkeypatch.setenv(flag, "true")
    searcher, _, _ = corpus(tmp_path)
    class HopModel(ScriptedModel):
        audits = 0
        async def generate(self, messages, **kwargs):
            result = await super().generate(messages, **kwargs)
            name = kwargs.get("response_format", {}).get("json_schema", {}).get("name")
            data=json.loads(result["content"])
            if name == "reading_plan":
                data["selected_docs"][0]["node_ids"]=["policy:0"]
            elif name == "evidence_verification":
                self.audits += 1
                if self.audits == 1:
                    data.update(sufficient=False, actions=[ReadAction(kind="read_nodes", doc_id="policy",
                                node_ids=["policy:29"]).model_dump()])
            elif name == "answer_structure":
                data["extracted_answer"]=data["extracted_answer"].replace("C1", "C2")
                data["supporting_claims"]=[s.replace("C1", "C2") for s in data["supporting_claims"]]
            elif name == "claim_grounding":
                data["claims"][0]["label"]="C2"
            return {**result,"content":json.dumps(data)}
    @contextmanager
    def span(*args, **kwargs):
        yield
    config=SimpleNamespace(parsed_dir=tmp_path/"parsed", settings={
        "agent":{"max_retrieval_loops":2},"prompt":{"version":"test"}})
    traces=TraceStore()
    service=AgenticReasoningService(config, HopModel(), PromptBuilder({}), AnswerFormatter(), traces,
                                    SimpleNamespace(span=span), wiki_searcher=searcher)
    response=await service.run(ChatRequest(query="Quasarneedle refund deadline"))
    assert response.grounded_claims[0]["node_id"] == "policy:29"
    agent=traces.get(response.trace_id).retrieval_params["agent"]
    assert len(agent["rounds"]) == 2
    assert agent["rounds"][0]["verification"]["sufficient"] is False
    assert agent["rounds"][1]["verification"]["sufficient"] is True
    assert "policy:29" in agent["tool_events"][0]["node_ids"]


def test_tree_navigation_pages_branches_and_expands_selected_path(tmp_path):
    searcher, _, _=corpus(tmp_path)
    tools=ReadTools(tmp_path/"parsed",searcher)
    _,event=tools.execute(ReadAction(kind="navigate_tree",doc_id="policy",offset=24,limit=6),"default")
    assert event["navigation"]["children"][-1]["path"] == ["Section 29"]
    _,branch=tools.execute(ReadAction(kind="navigate_tree",doc_id="policy",path=["Section 29"]),"default")
    assert [n["node_id"] for n in branch["navigation"]["nodes"]] == ["policy:29"]


@pytest.mark.asyncio
async def test_offline_retrieval_eval_requires_no_model_configuration(tmp_path, monkeypatch):
    import scripts.run_retrieval_eval as evaluation
    corpus(tmp_path)
    monkeypatch.delenv("GENERATION_MODEL_ALIAS", raising=False)
    monkeypatch.delenv("GENERATION_API_KEY", raising=False)
    monkeypatch.setenv("WIKI_OUTPUT_DIR", str(tmp_path/"wiki"))
    monkeypatch.setenv("PARSED_OUTPUT_DIR", str(tmp_path/"parsed"))
    def no_runtime_container():
        raise AssertionError("offline evaluation must not construct any model client")
    monkeypatch.setattr(evaluation.AppContainer, "from_env", no_runtime_container)
    dataset=tmp_path/"retrieval.jsonl"
    dataset.write_text(json.dumps({"query":"Quasarneedle refund", "expected_doc_ids":["policy"]})+"\n")
    args=evaluation._build_parser().parse_args(["--dataset",str(dataset),"--output",str(tmp_path/"report.json")])
    assert await evaluation._run(args) == 0


def test_global_version_catalog_overrides_stale_only_match(tmp_path):
    searcher, _, _ = corpus(tmp_path)
    newer = _doc("policy-new", "policy-family", "v2", "2026-01-01", "Updated unrelated vocabulary.")
    compiler = WikiCompiler(tmp_path/"wiki")
    compiler.upsert_document(newer, [Chunk(chunk_id="new:0", doc_id="policy-new", chunk_index=0,
        text=newer.content, source_path="new.md")])
    searcher.sync()
    d = AgenticDiscovery(config=SimpleNamespace(parsed_dir=tmp_path/"parsed", settings={}),
        wiki_searcher=searcher, generation_client=None, trace_store=None, tracing_manager=None)
    old_hits = searcher.search("Quasarneedle", 12, granularity="document")
    assert old_hits and old_hits[0].chunk.doc_id == "policy"
    current, _ = d._select_versions(old_hits)
    assert [h.chunk.doc_id for h in current] == ["policy-new"]
    historical, _ = d._select_versions(old_hits, filters={"effective_date_to":"2025-12-31"})
    assert historical[0].chunk.doc_id == "policy"
    compare, _ = d._select_versions(old_hits, filters={"_version_policy":"compare"})
    assert {h.chunk.doc_id for h in compare} == {"policy", "policy-new"}


def test_model_identity_rejects_second_model(tmp_path):
    cfg = AppConfig(tmp_path, {}, {"generation":{"default_alias":"one", "base_url":"https://same.example/v1"},
        "document_parser":{"enabled":True, "default_alias":"two", "base_url":"https://same.example"}}, {})
    with pytest.raises(ConfigurationError, match="single-model"):
        cfg.validate_model_identity()


@pytest.mark.asyncio
async def test_node_question_compilation_filters_unanchored_questions(tmp_path):
    _, doc, chunks = corpus(tmp_path, count=1)
    class CompilerModel:
        alias = "one"
        async def generate(self, messages, **kwargs):
            if kwargs.get("usage_stage") == "question_grounding":
                return {"content":json.dumps({"verdicts":[{"index":0,"answerable":True},
                                                           {"index":1,"answerable":False}]})}
            return {"content": json.dumps({"units":[{"node_id":chunks[0].chunk_id,
                "summary":"Refund deadline", "summary_quote":"refund deadline is 17 days",
                "aliases":["reimbursement"], "questions":[
                    {"question":"When to refund?", "answer_quote":"refund deadline is 17 days"},
                    {"question":"What fee?", "answer_quote":"refund deadline is 17 days"},
                    {"question":"What fee?", "answer_quote":"invented fee"}]}]})}
    result = await compile_knowledge(doc, chunks, CompilerModel())
    assert result["rejected_questions"] == 2
    assert chunks[0].metadata["synthetic_questions"] == ["When to refund?"]
    assert doc.metadata["knowledge_tree"][0]["sources"][0]["node_id"] == chunks[0].chunk_id


def test_table_aggregation_records_all_source_rows(tmp_path):
    searcher, doc, _ = corpus(tmp_path, count=1)
    rows=[]
    for i, amount in enumerate(["10.5", "20.5", "99"]):
        rows.append(Chunk(chunk_id=f"table:row:{i}", doc_id="policy", chunk_index=i,
            source_path=doc.source_path, text=f"Category=A; Amount USD={amount}",
            metadata={"kb_id":"default", "chunk_kind":"table_row", "table_node_id":"table",
                "table_headers":["Category", "Amount USD"],
                "table_columns":{"Category":0,"Amount USD":2},
                "table_row":{"0":"A" if i<2 else "B", "2":amount}}))
    _write_artifact(tmp_path/"parsed", "policy", doc, rows)
    tools = ReadTools(tmp_path/"parsed", searcher)
    action=ReadAction(kind="query_table", doc_id="policy", table_id="table", operation="sum",
        column="Amount USD", filters=[{"column":"Category", "op":"eq", "value":"A"}], limit=1)
    contexts, event=tools.execute(action, "default")
    assert event["value"] == "31.0"
    assert event["source_node_ids"] == ["table:row:0", "table:row:1"]
    assert contexts[-1]["calculation"]["matched_rows"] == 2
    with pytest.raises(ModelOutputError):
        tools.execute(action.model_copy(update={"column":"fabricated"}), "default")


@pytest.mark.asyncio
async def test_collection_map_reads_all_nodes_and_validates_quotes(tmp_path):
    searcher, _, chunks = corpus(tmp_path, count=2)
    class Mapper:
        async def generate(self, messages, **kwargs):
            batch=json.loads(messages[-1]["content"])
            return {"content":json.dumps({"facts":[{"node_id":c["chunk_id"],
                "quote":c["text"], "summary":"hint"} for c in batch]})}
    tools=ReadTools(tmp_path/"parsed", searcher, client=Mapper())
    contexts, scope=await tools.execute_async(ReadAction(kind="summarize_collection"), "default")
    assert {c["chunk_id"] for c in contexts} == {c.chunk_id for c in chunks}
    assert scope["complete"] and scope["contents_scanned"]
    assert all("map_fact" in c for c in contexts)


@pytest.mark.asyncio
async def test_collection_coverage_requires_visible_rows_and_mapped_facts(tmp_path):
    searcher, _, _ = corpus(tmp_path, count=2)
    class Mapper:
        async def generate(self, messages, **kwargs):
            return {"content": json.dumps({"facts":[{"node_id":c["chunk_id"], "quote":c["text"],
                "summary":"hint"} for c in json.loads(messages[-1]["content"]) ]})}
    tools = ReadTools(tmp_path/"parsed", searcher, client=Mapper())
    contexts, event = await tools.execute_async(ReadAction(kind="summarize_collection"), "default")
    service = AgenticReasoningService.__new__(AgenticReasoningService)
    state = {"query_plan":{"coverage_unit":"contents"}, "tool_events":[event],
             "evidence_set":EvidenceSet.build(contexts, EvidenceDriller(max_contexts=1), [])}
    coverage = service._collection_coverage(state)
    assert not coverage["complete"]
    assert coverage["map_facts_total"] == 2 and coverage["map_facts_visible"] == 1
    state["evidence_set"] = EvidenceSet.build(contexts, EvidenceDriller(), [])
    assert service._collection_coverage(state)["complete"]
    rows = [{"doc_id":"d", "chunk_id":f"row:{i}", "node_id":f"row:{i}",
             "citation_label":f"C{i+1}", "text":str(i), "source":"d"} for i in range(2)]
    first = {"kind":"query_table", "doc_id":"d", "table_id":"t", "filters":[],
             "operation":"rows", "source_node_ids":["row:0", "row:1"], "visible_node_ids":["row:0"]}
    state = {"query_plan":{"coverage_unit":"table_rows"}, "tool_events":[first],
             "evidence_set":EvidenceSet.build(rows, EvidenceDriller(), [])}
    assert not service._collection_coverage(state)["complete"]
    state["tool_events"].append({**first, "visible_node_ids":["row:1"]})
    assert service._collection_coverage(state)["complete"]
