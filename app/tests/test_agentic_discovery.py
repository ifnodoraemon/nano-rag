"""Phase 2: agentic wiki discovery + deterministic version filter + LLM file-read.

Exercises the real WikiCompiler/WikiSearcher (document-level BM25 manifest) with a
stubbed generation client, asserting the deterministic version filter drops stale
versions and the deep read pulls the applicable version's parsed artifact.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agentic import AgenticReasoningService
from app.agentic.discovery import AgenticDiscovery
from app.core.exceptions import ModelOutputError, RetrievalError
from app.core.tracing import TraceStore
from app.generation.answer_formatter import AnswerFormatter
from app.generation.prompt_builder import PromptBuilder
from app.schemas.chat import ChatRequest
from app.schemas.chunk import Chunk
from app.schemas.document import Document
from app.retrieval.hits import SearchHit
from app.wiki.compiler import WikiCompiler
from app.wiki.search import WikiSearcher


class FakeTracingManager:
    @contextmanager
    def span(self, name, attributes=None):  # noqa: ANN001
        yield


class _JsonReader:
    """Pulls the trailing ``Input JSON: {...}`` payload out of a message.

    Operates on the raw message content, NOT a re-serialization of the message
    list: ``json.dumps(messages)`` escapes the inner JSON's double quotes
    (``{"question"...`` -> ``{\\\"question\\\"...``), so a tail sliced from the
    re-serialized form is not valid JSON and would always raise — silently
    forcing the reading plan onto its degraded path and masking a real plan.
    Reading the content string directly keeps the embedded JSON intact.
    """

    marker = "Input JSON: "

    @classmethod
    def from_messages(cls, messages: list[dict]) -> dict:
        for message in messages:
            content = message.get("content")
            if not isinstance(content, str):
                continue
            start = content.find(cls.marker)
            if start < 0:
                continue
            return json.loads(content[start + len(cls.marker) :].strip())
        return {}


def _doc(doc_id: str, source_key: str, version: str, effective_date: str, body: str) -> Document:
    return Document(
        doc_id=doc_id,
        source_path=f"uploads/default/{doc_id}.md",
        title=f"{source_key.title()} {version}",
        content=body,
        metadata={
            "kb_id": "default",
            "source_key": source_key,
            "doc_type": "policy",
            "effective_date": effective_date,
            "version": version,
            "source_content_hash": f"hash-{doc_id}",
        },
    )


def _chunk(doc_id: str, section: str, text: str) -> Chunk:
    # A stable, per-chunk id derived from the content — the same scheme the real
    # ingester uses ({doc_id}:node:{hash}). Distinct sections must get distinct
    # ids: sharing one (e.g. ``{doc_id}:0`` for every section) would make the
    # context builder's dedup collapse distinct sections into a single context,
    # which previously masked a silently-degraded read plan.
    node = hashlib.sha256(f"{doc_id}|{section}|{text}".encode("utf-8")).hexdigest()[:8]
    return Chunk(
        chunk_id=f"{doc_id}:node:{node}",
        doc_id=doc_id,
        chunk_index=0,
        text=text,
        source_path=f"uploads/default/{doc_id}.md",
        title=section,
        metadata={"kb_id": "default", "hierarchy_path": [section]},
    )


def _write_artifact(parsed_dir: Path, doc_id: str, doc: Document, chunks: list[Chunk]) -> None:
    artifact = {
        "document": doc.model_dump(),
        "chunks": [chunk.model_dump() for chunk in chunks],
        "structured_document": None,
    }
    (parsed_dir / f"{doc_id}.json").write_text(
        json.dumps(artifact, ensure_ascii=False), encoding="utf-8"
    )


def _build_corpus(tmp_path: Path) -> tuple[WikiSearcher, Path]:
    wiki_dir = tmp_path / "wiki"
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    compiler = WikiCompiler(wiki_dir)

    v1 = _doc("travel-v1", "travel policy", "v1", "2023-01-01", "# Travel Policy\n\nOld rules.")
    v2 = _doc("travel-v2", "travel policy", "v2", "2025-06-01", "# Travel Policy\n\nNew rules.")
    compiler.upsert_document(v1, [_chunk("travel-v1", "General", "Old travel text.")])
    compiler.upsert_document(v2, [_chunk("travel-v2", "Reimbursement", "File expense claims within 15 days.")])

    _write_artifact(
        parsed_dir,
        "travel-v1",
        v1,
        [
            _chunk("travel-v1", "Reimbursement", "Old: file expense claims within 30 days."),
            _chunk("travel-v1", "Leave", "Old leave text."),
        ],
    )
    _write_artifact(
        parsed_dir,
        "travel-v2",
        v2,
        [
            _chunk("travel-v2", "Reimbursement", "New: file expense claims within 15 days."),
            _chunk("travel-v2", "Leave", "New leave text."),
        ],
    )
    return WikiSearcher(wiki_dir), parsed_dir


def _config(tmp_path: Path, **agent_overrides) -> SimpleNamespace:
    agent = {
        "max_retrieval_loops": 2,
        "max_subqueries": 4,
        "discovery_top_k": 12,
        "max_read_docs": 4,
        "max_read_chunks_per_doc": 24,
        **agent_overrides,
    }
    return SimpleNamespace(
        parsed_dir=tmp_path / "parsed",
        settings={"agent": agent, "retrieval": {"max_context_text_chars": 6000}},
    )


class ReadingPlanClient:
    """Returns a reading plan: read the latest-version candidate, optionally
    narrowed to given sections."""

    alias = "fake-gen"

    def __init__(self, focus_sections: list[str] | None = None) -> None:
        self.focus_sections = focus_sections

    async def generate(self, messages, **kwargs):  # noqa: ANN001
        rendered = json.dumps(messages, ensure_ascii=False)
        if "Decide which candidate documents" in rendered:
            payload = _JsonReader.from_messages(messages)
            candidates = payload.get("candidates", [])
            latest = [c for c in candidates if c.get("is_latest_version") is True]
            chosen = latest or candidates
            doc = (chosen[0] if chosen else {}).get("doc_id")
            return {
                "content": json.dumps(
                    {
                        "selected_docs": [
                            {
                                "doc_id": doc,
                                "focus_sections": self.focus_sections or [],
                                "reason": "latest applicable version",
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            }
        if "Break the user question" in rendered:
            return {"content": '{"subqueries":["travel reimbursement"]}', "finish_reason": "stop"}
        if "evidence auditor" in rendered:
            return {
                "content": '{"sufficient":true,"coverage_ratio":0.9,"missing_terms":[],"follow_up_queries":[],"reason":"closed"}'
            }
        return {
            "content": json.dumps(
                {
                    "is_answerable": True,
                    "missing_entities": [],
                    "extracted_answer": "File expense claims within 15 days. [C1]",
                    "supporting_claims": [
                        "[factual] The current travel policy sets a 15-day deadline. [C1]"
                    ],
                },
                ensure_ascii=False,
            ),
            "finish_reason": "stop",
        }


class BrokenPlanClient:
    alias = "broken"

    async def generate(self, messages, **kwargs):  # noqa: ANN001, ARG002
        raise RuntimeError("generation unavailable")


class HallucinatingPlanClient:
    """Answers the read plan, but every doc_id it returns is unknown to the
    candidate set. This is a contract violation: it must fail the request
    visibly — there is no degraded read-all fallback."""

    alias = "hallucinating"

    async def generate(self, messages, **kwargs):  # noqa: ANN001
        rendered = json.dumps(messages, ensure_ascii=False)
        if "Decide which candidate documents" in rendered:
            return {
                "content": json.dumps(
                    {
                        "selected_docs": [
                            {
                                "doc_id": "doc-that-does-not-exist",
                                "focus_sections": [],
                                "reason": "confabulated",
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            }
        if "Break the user question" in rendered:
            return {"content": '{"subqueries":["travel reimbursement"]}', "finish_reason": "stop"}
        if "evidence auditor" in rendered:
            return {
                "content": '{"sufficient":true,"coverage_ratio":0.9,"missing_terms":[],"follow_up_queries":[],"reason":"closed"}'
            }
        return {
            "content": json.dumps(
                {
                    "is_answerable": True,
                    "missing_entities": [],
                    "extracted_answer": "File expense claims within 15 days. [C1]",
                    "supporting_claims": [],
                },
                ensure_ascii=False,
            ),
            "finish_reason": "stop",
        }


class EmptyPlanClient:
    """Legitimately selects no documents: nothing is read, no fallback."""

    alias = "empty-plan"

    async def generate(self, messages, **kwargs):  # noqa: ANN001
        rendered = json.dumps(messages, ensure_ascii=False)
        if "Decide which candidate documents" in rendered:
            return {"content": '{"selected_docs": []}', "finish_reason": "stop"}
        if "Break the user question" in rendered:
            return {"content": '{"subqueries":["travel reimbursement"]}', "finish_reason": "stop"}
        if "evidence auditor" in rendered:
            return {
                "content": '{"sufficient":false,"coverage_ratio":0.0,"missing_terms":["deadline"],"follow_up_queries":[],"reason":"no_docs_read"}'
            }
        return {
            "content": json.dumps(
                {
                    "is_answerable": False,
                    "missing_entities": ["travel reimbursement"],
                    "extracted_answer": "",
                    "supporting_claims": [],
                },
                ensure_ascii=False,
            ),
            "finish_reason": "stop",
        }


def _hit(doc_id: str, score: float, source_key: str | None = None) -> SearchHit:
    """A minimal SearchHit for unit-testing the version filter and plan parsing
    without a wiki corpus. Control the score and optional source_key directly."""
    metadata = {"kb_id": "default"}
    if source_key is not None:
        metadata["source_key"] = source_key
    chunk = Chunk(
        chunk_id=f"{doc_id}:node:0",
        doc_id=doc_id,
        chunk_index=0,
        text=f"{doc_id} body",
        source_path=f"uploads/default/{doc_id}.md",
        title=doc_id,
        metadata=metadata,
    )
    return SearchHit(chunk=chunk, score=score)


def _discovery(config, searcher, client) -> AgenticDiscovery:
    return AgenticDiscovery(
        config=config,
        wiki_searcher=searcher,
        generation_client=client,
        trace_store=TraceStore(),
        tracing_manager=FakeTracingManager(),
    )


@pytest.mark.asyncio
async def test_discovery_filters_latest_version_and_reads_parsed(tmp_path: Path) -> None:
    searcher, parsed_dir = _build_corpus(tmp_path)
    config = _config(tmp_path)
    config.parsed_dir = parsed_dir
    discovery = AgenticDiscovery(
        config=config,
        wiki_searcher=searcher,
        generation_client=ReadingPlanClient(),
        trace_store=TraceStore(),
        tracing_manager=FakeTracingManager(),
    )

    contexts, trace = await discovery.retrieve(ChatRequest(query="travel reimbursement deadline"), "travel reimbursement deadline")

    # Version filter: v1 (stale) dropped, v2 (latest) kept.
    groups = trace["retrieval_params"]["version_filter"]["groups"]
    assert groups and groups[0]["winner"] == "travel-v2"
    assert "travel-v1" in groups[0]["dropped"]
    # Deep read came from the v2 artifact only.
    read_docs = trace["retrieval_params"]["read_doc_ids"]
    assert read_docs == ["travel-v2"]
    assert contexts
    assert "New: file expense claims within 15 days." in " ".join(
        str(c.get("text")) for c in contexts
    )
    # The latest version is tagged primary evidence.
    assert any(c.get("is_latest_version") is True for c in contexts)


@pytest.mark.asyncio
async def test_discovery_focus_sections_narrow_the_read(tmp_path: Path) -> None:
    searcher, parsed_dir = _build_corpus(tmp_path)
    config = _config(tmp_path)
    config.parsed_dir = parsed_dir
    discovery = AgenticDiscovery(
        config=config,
        wiki_searcher=searcher,
        generation_client=ReadingPlanClient(focus_sections=["Reimbursement"]),
        trace_store=TraceStore(),
        tracing_manager=FakeTracingManager(),
    )

    contexts, _ = await discovery.retrieve(ChatRequest(query="travel reimbursement"), "travel reimbursement")

    # Only the Reimbursement chunk of the selected doc is read, not Leave.
    texts = " ".join(str(c.get("text")) for c in contexts)
    assert "New: file expense claims within 15 days." in texts
    assert "New leave text." not in texts


@pytest.mark.asyncio
async def test_discovery_fails_loud_when_plan_llm_unavailable(tmp_path: Path) -> None:
    # No degraded read-all: a gateway failure during the reading plan raises
    # and fails the request visibly.
    searcher, parsed_dir = _build_corpus(tmp_path)
    config = _config(tmp_path)
    config.parsed_dir = parsed_dir
    discovery = AgenticDiscovery(
        config=config,
        wiki_searcher=searcher,
        generation_client=BrokenPlanClient(),
        trace_store=TraceStore(),
        tracing_manager=FakeTracingManager(),
    )

    with pytest.raises(RuntimeError, match="generation unavailable"):
        await discovery.retrieve(ChatRequest(query="travel reimbursement"), "travel reimbursement")


@pytest.mark.asyncio
async def test_discovery_empty_plan_reads_nothing(tmp_path: Path) -> None:
    # An empty selection is a legitimate structured decision — no docs are
    # read and there is no fallback to reading the top candidates.
    searcher, parsed_dir = _build_corpus(tmp_path)
    config = _config(tmp_path)
    config.parsed_dir = parsed_dir
    discovery = AgenticDiscovery(
        config=config,
        wiki_searcher=searcher,
        generation_client=EmptyPlanClient(),
        trace_store=TraceStore(),
        tracing_manager=FakeTracingManager(),
    )

    contexts, trace = await discovery.retrieve(ChatRequest(query="travel reimbursement"), "travel reimbursement")

    assert contexts == []
    assert trace["retrieval_params"]["read_doc_ids"] == []
    assert trace["retrieval_params"]["reading_plan"] == {"selected_docs": []}


@pytest.mark.asyncio
async def test_discovery_select_versions_is_deterministic(tmp_path: Path) -> None:
    searcher, parsed_dir = _build_corpus(tmp_path)
    config = _config(tmp_path)
    config.parsed_dir = parsed_dir
    discovery = AgenticDiscovery(
        config=config,
        wiki_searcher=searcher,
        generation_client=ReadingPlanClient(),
        trace_store=TraceStore(),
        tracing_manager=FakeTracingManager(),
    )
    discovered = await discovery._discover("travel reimbursement", "default", None)
    candidates, report = discovery._select_versions(discovered)
    # Both versions are discovered by BM25; the deterministic filter keeps v2.
    assert {hit.chunk.doc_id for hit in discovered} == {"travel-v1", "travel-v2"}
    assert [hit.chunk.doc_id for hit in candidates] == ["travel-v2"]
    assert report["groups"][0]["winner"] == "travel-v2"


def test_select_versions_orders_candidates_by_discovery_score() -> None:
    """M-1: the survivor list handed to the LLM / degraded read-all must be in
    score order, not 'ungrouped first, then winners in first-seen order'. A
    low-scoring ungrouped doc must not displace a higher-scoring winner."""
    discovery = _discovery(
        SimpleNamespace(parsed_dir=Path("/tmp"), settings={}),
        None,
        None,
    )
    discovered = [
        _hit("ungrouped-low", 0.5),  # no source_key
        _hit("winner-a", 0.9, "policy a"),
        _hit("winner-b", 0.8, "policy b"),
        _hit("winner-c", 0.7, "policy c"),
        _hit("winner-d", 0.6, "policy d"),
    ]
    candidates, _ = discovery._select_versions(discovered)
    assert [hit.chunk.doc_id for hit in candidates] == [
        "winner-a",
        "winner-b",
        "winner-c",
        "winner-d",
        "ungrouped-low",
    ]


@pytest.mark.asyncio
async def test_hallucinated_plan_fails_loud(tmp_path: Path) -> None:
    """A plan whose doc_ids are unknown to the candidate set is a contract
    violation: it must raise, not silently fall back to a bounded read-all."""
    searcher, parsed_dir = _build_corpus(tmp_path)
    config = _config(tmp_path)
    config.parsed_dir = parsed_dir
    discovery = _discovery(config, searcher, HallucinatingPlanClient())

    with pytest.raises(ModelOutputError, match="unknown doc_id"):
        await discovery.retrieve(
            ChatRequest(query="travel reimbursement"), "travel reimbursement"
        )


def test_parse_plan_caps_at_max_read_docs(tmp_path: Path) -> None:
    """Pin the _parse_plan invariants — duplicate doc_ids deduped and the
    result capped at max_read_docs. The cap is set to 2 so it actually
    truncates the three valid docs."""
    discovery = _discovery(_config(tmp_path, max_read_docs=2), None, None)
    candidates = [
        _hit(f"doc-{i}", float(1 - i * 0.05)) for i in range(6)
    ]
    parsed = {
        "selected_docs": [
            {"doc_id": "doc-0", "focus_sections": [], "reason": "r"},
            {"doc_id": "doc-1", "focus_sections": [], "reason": "r"},
            {"doc_id": "doc-1", "focus_sections": [], "reason": "dup"},  # duplicate
            {"doc_id": "doc-2", "focus_sections": [], "reason": "r"},
        ]
    }
    plan = discovery._parse_plan(parsed, candidates)
    doc_ids = [entry["doc_id"] for entry in plan["selected_docs"]]
    assert doc_ids.count("doc-1") == 1  # deduped
    assert len(doc_ids) == discovery.max_read_docs  # capped at 2
    assert doc_ids == ["doc-0", "doc-1"]  # doc-2 dropped by the cap


def test_parse_plan_rejects_hallucinated_doc_id(tmp_path: Path) -> None:
    discovery = _discovery(_config(tmp_path), None, None)
    candidates = [_hit(f"doc-{i}", 1.0) for i in range(3)]
    parsed = {
        "selected_docs": [
            {"doc_id": "doc-99", "focus_sections": [], "reason": "hallucination"},
        ]
    }
    with pytest.raises(ModelOutputError, match="unknown doc_id"):
        discovery._parse_plan(parsed, candidates)


@pytest.mark.asyncio
async def test_unresolvable_focus_sections_fail_loud(tmp_path: Path) -> None:
    """Focus sections that match no heading must raise instead of silently
    reading the whole document."""
    searcher, parsed_dir = _build_corpus(tmp_path)
    config = _config(tmp_path)
    config.parsed_dir = parsed_dir
    discovery = _discovery(
        config, searcher, ReadingPlanClient(focus_sections=["No Such Section"])
    )

    with pytest.raises(ModelOutputError, match="do not match any section"):
        await discovery.retrieve(
            ChatRequest(query="travel reimbursement"), "travel reimbursement"
        )


@pytest.mark.asyncio
async def test_missing_parsed_artifact_fails_loud(tmp_path: Path) -> None:
    """A discovery page whose parsed artifact is gone is a data-integrity
    failure: raise instead of silently skipping the document."""
    searcher, parsed_dir = _build_corpus(tmp_path)
    (parsed_dir / "travel-v2.json").unlink()
    config = _config(tmp_path)
    config.parsed_dir = parsed_dir
    discovery = _discovery(config, searcher, ReadingPlanClient())

    with pytest.raises(RetrievalError, match="missing"):
        await discovery.retrieve(
            ChatRequest(query="travel reimbursement"), "travel reimbursement"
        )


@pytest.mark.asyncio
async def test_service_uses_wiki_discovery(tmp_path: Path) -> None:
    """M-3: with wiki_searcher wired, the service's agentic branch of
    _retrieve delegates to AgenticDiscovery and labels the trace
    retrieval_engine == 'agentic_wiki'. The wiki layer is the only retrieval
    engine in this build — there is no dense fallback to fall back to."""
    searcher, parsed_dir = _build_corpus(tmp_path)
    config = _config(tmp_path)
    config.parsed_dir = parsed_dir
    config.settings = {
        "agent": {
            "max_retrieval_loops": 2,
            "max_subqueries": 4,
            "max_context_text_chars": 6000,
        },
        "prompt": {"version": "test"},
    }
    trace_store = TraceStore()
    service = AgenticReasoningService(
        config=config,
        generation_client=ReadingPlanClient(),
        prompt_builder=PromptBuilder({"chat": {"system": "system"}}),
        answer_formatter=AnswerFormatter(),
        trace_store=trace_store,
        tracing_manager=FakeTracingManager(),
        wiki_searcher=searcher,
    )
    assert service.agentic_discovery is not None

    response = await service.run(ChatRequest(query="travel reimbursement"))

    # The wiki path handled retrieval; the answer came from the latest
    # version's parsed artifact.
    assert "15 days" in response.answer
    record = trace_store.get(response.trace_id)
    assert record is not None
    assert record.retrieval_params["agent"]["engine"] == "langgraph"
    assert record.retrieval_params["agent"]["retrieval_engine"] == "agentic_wiki"
    # The discovery trace is present and marked the winner as primary.
    assert record.retrieval_params.get("read_doc_ids") == ["travel-v2"]
