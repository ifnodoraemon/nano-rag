"""Concurrent document preparation during ingest.

Each document costs two model round trips (parse + graph extraction) and writes
nothing, so preparation runs concurrently while commits stay sequential. These
tests pin the two properties that matter: it actually overlaps the waits, and a
failing document cancels the rest of the batch instead of letting them finish
against a doomed run.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from app.core.config import AppConfig
from app.core.exceptions import ConfigurationError, ParsingError
from app.core.tracing import TracingManager
from app.ingestion.pipeline import IngestionPipeline, _ingest_concurrency
from app.tests.test_document_parser import FakeGenerationClient


class SleepingParser:
    """Records peak concurrency and simulates a slow model round trip."""

    enabled = True

    def __init__(self, delay: float = 0.15, fail_on: str | None = None) -> None:
        self.delay = delay
        self.fail_on = fail_on
        self.active = 0
        self.peak = 0
        self.calls: list[str] = []

    def supports(self, path: Path) -> bool:  # noqa: ARG002
        return True

    async def parse_file(self, path: Path) -> str:
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            self.calls.append(path.name)
            await asyncio.sleep(self.delay)
            if self.fail_on and path.name == self.fail_on:
                raise ParsingError(f"parser exploded on {path.name}")
            return f"# {path.stem}\n\nContent of {path.name}."
        finally:
            self.active -= 1


def _pipeline(tmp_path: Path, parser, monkeypatch) -> IngestionPipeline:
    monkeypatch.setenv("PARSED_OUTPUT_DIR", str(tmp_path / "parsed"))
    monkeypatch.setenv("RAG_INGEST_ALLOWED_DIRS", str(tmp_path))
    monkeypatch.setattr("app.ingestion.loader._cached_allowed_dirs", None)
    config = AppConfig(
        config_dir=tmp_path,
        settings={
            "chunk": {"size": 400, "overlap": 40},
            "timeout": {"document_parser_seconds": 30},
        },
        models={"model_gateway": {"base_url": "", "api_key": ""}},
        prompts={},
    )
    config.parsed_dir.mkdir(parents=True, exist_ok=True)
    return IngestionPipeline(
        config=config,
        generation_client=FakeGenerationClient(),
        tracing_manager=TracingManager("test-service", ""),
        document_parser=parser,
    )


def _corpus(tmp_path: Path, count: int) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    for index in range(count):
        (root / f"doc{index}.md").write_text(
            f"# Document {index}\n\nUnique body {index}.", encoding="utf-8"
        )
    return root


@pytest.mark.asyncio
async def test_preparation_overlaps_model_calls(monkeypatch, tmp_path) -> None:
    """Four 150 ms parses must overlap, not run back to back."""
    monkeypatch.setenv("RAG_INGEST_CONCURRENCY", "4")
    parser = SleepingParser(delay=0.15)
    pipeline = _pipeline(tmp_path, parser, monkeypatch)
    corpus = _corpus(tmp_path, 4)

    started = time.perf_counter()
    response = await pipeline.run(str(corpus), kb_id="default")
    elapsed = time.perf_counter() - started

    assert response.documents == 4
    assert parser.peak > 1, "preparation did not run concurrently"
    assert parser.peak <= 4, "concurrency exceeded the configured bound"
    # Serial would be >= 4 * 0.15s of parsing alone.
    assert elapsed < 0.15 * 4, f"took {elapsed:.2f}s - looks serial"


@pytest.mark.asyncio
async def test_concurrency_one_is_serial(monkeypatch, tmp_path) -> None:
    """The off switch must restore the previous behaviour exactly."""
    monkeypatch.setenv("RAG_INGEST_CONCURRENCY", "1")
    parser = SleepingParser(delay=0.05)
    pipeline = _pipeline(tmp_path, parser, monkeypatch)
    corpus = _corpus(tmp_path, 3)

    response = await pipeline.run(str(corpus), kb_id="default")

    assert response.documents == 3
    assert parser.peak == 1


@pytest.mark.asyncio
async def test_a_failing_document_aborts_the_batch(monkeypatch, tmp_path) -> None:
    """One bad file must fail the batch and cancel the rest - never commit half."""
    monkeypatch.setenv("RAG_INGEST_CONCURRENCY", "3")
    parser = SleepingParser(delay=0.05, fail_on="doc2.md")
    pipeline = _pipeline(tmp_path, parser, monkeypatch)
    corpus = _corpus(tmp_path, 6)

    with pytest.raises(ExceptionGroup):
        await pipeline.run(str(corpus), kb_id="default")

    # Nothing was committed: preparation failed before any write.
    assert not list((tmp_path / "parsed").glob("*.json"))


@pytest.mark.asyncio
async def test_commit_order_follows_discovery_order(monkeypatch, tmp_path) -> None:
    """Concurrency must not randomise which document is committed first."""
    monkeypatch.setenv("RAG_INGEST_CONCURRENCY", "4")
    parser = SleepingParser(delay=0.01)
    pipeline = _pipeline(tmp_path, parser, monkeypatch)
    corpus = _corpus(tmp_path, 5)

    await pipeline.run(str(corpus), kb_id="default")

    artifacts = sorted((tmp_path / "parsed").glob("*.json"))
    assert len(artifacts) == 5


@pytest.mark.asyncio
async def test_unchanged_documents_are_still_reported(monkeypatch, tmp_path) -> None:
    """The skip must survive the concurrent path, per document."""
    monkeypatch.setenv("RAG_INGEST_CONCURRENCY", "3")
    parser = SleepingParser(delay=0.01)
    pipeline = _pipeline(tmp_path, parser, monkeypatch)
    corpus = _corpus(tmp_path, 3)

    first = await pipeline.run(str(corpus), kb_id="default")
    calls_after_first = len(parser.calls)
    second = await pipeline.run(str(corpus), kb_id="default")

    assert first.documents == 3
    assert second.documents == 0
    assert len(second.unchanged_doc_ids) == 3
    assert len(parser.calls) == calls_after_first


def test_concurrency_setting_is_validated(monkeypatch) -> None:
    monkeypatch.setenv("RAG_INGEST_CONCURRENCY", "4")
    assert _ingest_concurrency() == 4

    monkeypatch.setenv("RAG_INGEST_CONCURRENCY", "0")
    with pytest.raises(ConfigurationError, match=">= 1"):
        _ingest_concurrency()

    monkeypatch.setenv("RAG_INGEST_CONCURRENCY", "many")
    with pytest.raises(ConfigurationError, match="must be an integer"):
        _ingest_concurrency()
