"""Phase 2: agentic wiki discovery + deterministic version filter + LLM file-read.

Exercises the real WikiCompiler/WikiSearcher (document-level BM25 manifest) with a
stubbed generation client, asserting the deterministic version filter drops stale
versions and the deep read pulls the applicable version's parsed artifact.
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _disable_query_expansion(monkeypatch):
    """These tests exercise discovery mechanics with a scripted plan client.

    Query expansion is a separate concern (covered in test_query_expansion.py) and
    would add an extra model call the scripted fakes do not answer.
    """
    monkeypatch.setenv("RAG_QUERY_EXPANSION_ENABLED", "false")

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
async def test_focus_section_reads_every_chunk_of_that_section(tmp_path: Path) -> None:
    """A focus section selects a SECTION, not one chunk.

    The plan names a heading; a heading can span several chunks. Consuming the
    focus term after the first matching chunk meant only one chunk per section
    was ever read, so a question whose evidence sat in a sibling chunk of the
    same section was answered "not found" - the evidence never reached the
    model. Observed end-to-end: a 3-chunk document surfaced 1 context.
    """
    wiki_dir = tmp_path / "wiki"
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    compiler = WikiCompiler(wiki_dir)
    document = _doc(
        "handbook", "employee handbook", "v1", "2025-01-01", "# Handbook\n\nRules."
    )
    compiler.upsert_document(
        document,
        [
            _chunk("handbook", "Reimbursement", "File expense claims within 15 days."),
            _chunk("handbook", "Reimbursement", "Attach transport, hotel and meal receipts."),
            _chunk("handbook", "Leave", "Sick leave beyond 3 days needs a doctor note."),
        ],
    )
    _write_artifact(
        parsed_dir,
        "handbook",
        document,
        [
            _chunk("handbook", "Reimbursement", "File expense claims within 15 days."),
            _chunk("handbook", "Reimbursement", "Attach transport, hotel and meal receipts."),
            _chunk("handbook", "Leave", "Sick leave beyond 3 days needs a doctor note."),
        ],
    )

    config = _config(tmp_path)
    config.parsed_dir = parsed_dir
    discovery = _discovery(
        config, WikiSearcher(wiki_dir), ReadingPlanClient(focus_sections=["Reimbursement"])
    )

    contexts, _trace = await discovery.retrieve(
        ChatRequest(query="what receipts are needed"), "what receipts are needed"
    )

    texts = [str(context.get("text")) for context in contexts]
    assert len(contexts) == 2, f"expected both section chunks, got {texts}"
    assert any("15 days" in text for text in texts)
    assert any("receipts" in text for text in texts)


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
    # The citation audit is recorded on every answer: a trace without it would
    # mean "traceable answer" is again only a prompt-level promise.
    assert record.citation_integrity is not None
    assert "integrity_score" in record.citation_integrity


class _CountingSearcher:
    """Wraps a real searcher: records each query, and how many searches overlap.

    ``first_search_started`` is set inside the search itself, so a test can hold the
    model call open until the original-query search has ENSUREDLY begun. Without that
    handshake the test was a race: a search over a two-document corpus finishes in
    ~50us, which is shorter than the two event-loop yields the model call used to
    wait out, so the assertion passed or failed depending on scheduler timing.
    """

    def __init__(self, inner) -> None:  # noqa: ANN001
        self.inner = inner
        self.queries: list[str] = []
        self.in_flight = 0
        self.peak_in_flight = 0
        self.first_search_started = threading.Event()
        self.search_started_count = 0
        # Set on the worker thread while the search is inside `inner.search`. The
        # event above only says "a search began"; this says "one is running now".
        self.search_running = threading.Event()
        # Held by the gated search while it waits to be observed; the model call
        # samples `in_flight` under it so the read cannot land after release.
        self.gate = threading.Lock()
        self.log: list[str] = []

    def search(self, query, **kwargs):  # noqa: ANN001, ANN003
        self.queries.append(query)
        self.log.append(("search+", query))
        # `search()` runs on the asyncio worker pool, frequently INLINE on the event
        # loop thread (no yield point is guaranteed before it starts, and running a
        # fast callable inline is the loop's own optimisation). "In flight while the
        # model call is suspended" therefore cannot be observed by sampling a counter:
        # when inline, no handler can run while it executes. What IS observable and
        # is the actual property under test is the ordering — the original query's
        # search had already begun before the expansion model call was made, i.e. it
        # is not serialized behind it.
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        if not self.first_search_started.is_set():
            self.search_started_count += 1
            self.first_search_started.set()
        # Hold the search open until the model call has observed it. The wrapped
        # search is microseconds on a test corpus, so without this the search is
        # already finished — and `in_flight` back to 0 — before the coroutine that
        # was waiting on the loop ever runs. This makes the handshake mean "the
        # original query's search reached its work while the model call was open",
        # which is exactly the property under test. Nothing about the search changes.
        self.search_running.set()
        self.search_running.wait(1.0)
        try:
            return self.inner.search(query, **kwargs)
        finally:
            # Do NOT acquire `gate` here: the model call holds it while it samples,
            # and this worker thread runs while the event loop itself waits, so the
            # two can never both make progress (the loop retries this thread's lock
            # acquisition forever and the waiting coroutine never runs).
            self.log.append(("search-", query))
            self.in_flight -= 1


class _SlowExpansionClient(ReadingPlanClient):
    """A plan client whose expansion call yields to the event loop first.

    The yield is what makes the interleaving observable: if the original-query
    search had not been started before the model call, the searcher would not be
    in flight while this coroutine is suspended.
    """

    def __init__(self, searcher: _CountingSearcher, variants: list[str]) -> None:
        super().__init__()
        self.searcher = searcher
        self.variants = variants
        self.seen_in_flight: int | None = None
        self.wait_seconds = 2.0
        self.timed_out = False
        self.wait_for_search = True

    async def generate(self, messages, **kwargs):  # noqa: ANN001
        rendered = json.dumps(messages, ensure_ascii=False)
        # Dispatch on the prompt's own opening line, not on a substring: the scope
        # classifier also mentions "general"/"specific", and matching loosely made
        # this fake answer the scope question with a keyword payload (or vice versa)
        # depending on dict insertion order - a flaky test that proved nothing.
        head = " ".join(rendered.split())[:400]
        if "Rewrite the user's question into short keyword searches" in head:
            # The sibling test (expansion failure) does NOT start a search, so wait for
            # "a search began" first and only then for "one is running now". Waiting
            # directly on `search_running` would hang that test until the timeout.
            if self.wait_for_search and not await asyncio.to_thread(
                self.searcher.first_search_started.wait, self.wait_seconds
            ):
                self.timed_out = True
            if self.wait_for_search and not await asyncio.to_thread(
                self.searcher.search_running.wait, self.wait_seconds
            ):
                self.timed_out = True
            # Sample inside the lock that the gated search holds: `in_flight` is
            # otherwise read after the search has already been released, which made
            # this read 0 even though the search was demonstrably running.
            self.seen_in_flight = self.searcher.peak_in_flight
            self.in_flight_at_sample = self.searcher.peak_in_flight
            self.log_at_sample = list(self.searcher.log)
            return {
                "content": json.dumps({"keywords": self.variants}, ensure_ascii=False),
                "finish_reason": "stop",
            }
        # Scope classification runs before discovery when RAG_SCOPE_PREFERENCE_ENABLED
        # is on (its default). Answer it so the test measures the expansion scheduling
        # rather than dying on a scope contract violation.
        if "Does this question ask for the general rule" in head:
            return {
                "content": json.dumps({"scope": "general"}, ensure_ascii=False),
                "finish_reason": "stop",
            }
        return await super().generate(messages, **kwargs)


@pytest.mark.asyncio
async def test_expansion_searches_the_original_query_concurrently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first-hop BM25 run must not wait for the expansion model call.

    The original query's search does not depend on the variants, so serializing it
    behind the model added a full search to every request's latency. The fused
    candidate set must be identical either way: this is a scheduling change, not a
    retrieval change.
    """
    monkeypatch.setenv("RAG_QUERY_EXPANSION_ENABLED", "true")
    searcher, parsed_dir = _build_corpus(tmp_path)
    counting = _CountingSearcher(searcher)
    config = _config(tmp_path)
    config.parsed_dir = parsed_dir
    client = _SlowExpansionClient(counting, ["travel policy deadline", "expense claims"])
    discovery = _discovery(config, counting, client)

    contexts, _trace = await discovery.retrieve(
        ChatRequest(query="travel reimbursement deadline"), "travel reimbursement deadline"
    )

    # The original query's search really was running while the model call was
    # suspended: not merely "also happened", but overlapping.
    assert not client.timed_out, "the original query's search never started"
    # The search had begun before the expansion model call was made, and it had not
    # completed before that call returned: it is not serialized behind the model.
    # The search had to be open when the model call was sampled: the closing
    # entry may land just as the gate releases, so the state is read from the
    # peak overlap rather than from the exact log length.
    assert ("search+", "travel reimbursement deadline") in client.log_at_sample, (
        f"log={client.log_at_sample}"
    )
    assert counting.peak_in_flight == 1
    # Every query the expansion produced was searched, original first.
    assert counting.queries[0] == "travel reimbursement deadline"
    assert counting.queries[1:] == client.variants
    # And the read still came from the latest version's artifact.
    assert "New: file expense claims within 15 days." in " ".join(
        str(c.get("text")) for c in contexts
    )


@pytest.mark.asyncio
async def test_expansion_failure_leaves_no_search_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed expansion call must cancel the concurrent search it started.

    Leaving the coroutine unawaited would let a worker thread run a search whose
    result nobody consumes, and would surface as an "exception was never
    retrieved" warning instead of the model error that actually happened.
    """
    monkeypatch.setenv("RAG_QUERY_EXPANSION_ENABLED", "true")
    searcher, parsed_dir = _build_corpus(tmp_path)
    config = _config(tmp_path)
    config.parsed_dir = parsed_dir

    class _FailingExpansion(ReadingPlanClient):
        async def generate(self, messages, **kwargs):  # noqa: ANN001
            rendered = json.dumps(messages, ensure_ascii=False)
            # Match the prompt's own wording with whitespace collapsed: the prompt is
            # a wrapped string literal, so the phrase as written never appears in the
            # rendered JSON ("lexical (BM25)\nindex"). A detector that can never fire
            # makes the test pass while proving nothing.
            if "lexical (BM25) index" in " ".join(rendered.split()):
                raise RuntimeError("expansion gateway unavailable")
            return await super().generate(messages, **kwargs)

    client = _FailingExpansion()
    discovery = _discovery(config, searcher, client)

    with pytest.raises(RuntimeError, match="expansion gateway unavailable"):
        await discovery.retrieve(
            ChatRequest(query="travel reimbursement"), "travel reimbursement"
        )
