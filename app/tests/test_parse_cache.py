"""Parse caching and the unchanged-document fast path.

Parsing runs through the multimodal model for every file, so an ingest that
cannot prove the result would be identical must do the work again - but one that
can prove it must not spend a model call reproducing an artifact already on disk.

Two distinct mechanisms are covered here:

- the content-addressed parse cache (same bytes, any path);
- the unchanged-document skip (same path, same bytes, same configuration).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.config import AppConfig
from app.core.exceptions import StoreError
from app.ingestion.parse_cache import ParseCache, build_fingerprint, content_hash
from app.ingestion.parser_registry import parse_content
from app.tests.test_document_parser import (
    EmptyDocumentParser,
    FakeDocumentParser,
    FakeGenerationClient,
    _parsed_chunks,
)
from app.core.tracing import TracingManager
from app.ingestion.pipeline import IngestionPipeline


def _cache(tmp_path: Path, fingerprint: str | None = None, *, enabled: bool = True) -> ParseCache:
    return ParseCache(
        tmp_path / "parsed",
        fingerprint
        or build_fingerprint(
            parser_provider="qwen",
            parser_model="vl-model",
            chunk_size=400,
            chunk_overlap=40,
        ),
        enabled=enabled,
    )


class CountingParser(FakeDocumentParser):
    """Counts model calls so a cache hit is measurable, not inferred."""

    def __init__(self) -> None:
        self.calls = 0

    async def parse_file(self, path: Path) -> str:
        self.calls += 1
        return await super().parse_file(path)


@pytest.mark.asyncio
async def test_parse_cache_avoids_a_second_model_call(tmp_path: Path) -> None:
    document = tmp_path / "policy.md"
    document.write_text("# Policy\n\nSubmit within 15 days.\n", encoding="utf-8")
    cache = _cache(tmp_path)
    parser = CountingParser()

    first = await parse_content(document, parser, cache)
    second = await parse_content(document, parser, cache)

    assert parser.calls == 1, "identical bytes must be parsed once"
    assert first.metadata["parse_cache"] == "miss"
    assert second.metadata["parse_cache"] == "hit"
    assert second.text == first.text


@pytest.mark.asyncio
async def test_parse_cache_hit_survives_a_path_change(tmp_path: Path) -> None:
    """Content-addressed, not path-addressed: a moved file is still a hit."""
    original = tmp_path / "policy.md"
    original.write_text("# Policy\n\nSubmit within 15 days.\n", encoding="utf-8")
    moved = tmp_path / "renamed.md"
    moved.write_bytes(original.read_bytes())

    cache = _cache(tmp_path)
    parser = CountingParser()
    await parse_content(original, parser, cache)
    result = await parse_content(moved, parser, cache)

    assert parser.calls == 1
    assert result.metadata["parse_cache"] == "hit"


@pytest.mark.asyncio
async def test_parse_cache_misses_when_content_changes(tmp_path: Path) -> None:
    document = tmp_path / "policy.md"
    document.write_text("# Policy\n\nVersion one.\n", encoding="utf-8")
    cache = _cache(tmp_path)
    parser = CountingParser()

    await parse_content(document, parser, cache)
    document.write_text("# Policy\n\nVersion two.\n", encoding="utf-8")
    await parse_content(document, parser, cache)

    assert parser.calls == 2


@pytest.mark.asyncio
async def test_parse_cache_is_namespaced_by_configuration(tmp_path: Path) -> None:
    """A different parser model must not be served the previous model's output."""
    document = tmp_path / "policy.md"
    document.write_text("# Policy\n\nText.\n", encoding="utf-8")
    parser = CountingParser()

    small = _cache(tmp_path, build_fingerprint(
        parser_provider="qwen", parser_model="model-a", chunk_size=400, chunk_overlap=40
    ))
    large = _cache(tmp_path, build_fingerprint(
        parser_provider="qwen", parser_model="model-b", chunk_size=400, chunk_overlap=40
    ))

    await parse_content(document, parser, small)
    second = await parse_content(document, parser, large)

    assert parser.calls == 2, "switching models must re-parse"
    assert second.metadata["parse_cache"] == "miss"


@pytest.mark.asyncio
async def test_parse_cache_disabled_always_parses(tmp_path: Path) -> None:
    document = tmp_path / "policy.md"
    document.write_text("# Policy\n\nText.\n", encoding="utf-8")
    parser = CountingParser()
    cache = _cache(tmp_path, enabled=False)

    await parse_content(document, parser, cache)
    second = await parse_content(document, parser, cache)

    assert parser.calls == 2
    assert second.metadata["parse_cache"] == "disabled"


@pytest.mark.asyncio
async def test_corrupt_cache_entry_fails_loudly(tmp_path: Path) -> None:
    """A torn cache file is a disk problem, not a cache miss."""
    document = tmp_path / "policy.md"
    document.write_text("# Policy\n\nText.\n", encoding="utf-8")
    cache = _cache(tmp_path)
    digest = content_hash(document)

    await parse_content(document, FakeDocumentParser(), cache)
    cache._path(digest).write_text("{not json", encoding="utf-8")  # noqa: SLF001

    with pytest.raises(StoreError, match="unreadable"):
        await parse_content(document, FakeDocumentParser(), cache)


@pytest.mark.asyncio
async def test_empty_cache_entry_fails_loudly(tmp_path: Path) -> None:
    document = tmp_path / "policy.md"
    document.write_text("# Policy\n\nText.\n", encoding="utf-8")
    cache = _cache(tmp_path)
    digest = content_hash(document)
    cache.put(digest, text="", parser_name="x")

    with pytest.raises(StoreError, match="no usable text"):
        await parse_content(document, FakeDocumentParser(), cache)


def _pipeline(tmp_path: Path, parser, monkeypatch) -> IngestionPipeline:
    monkeypatch.setenv("PARSED_OUTPUT_DIR", str(tmp_path / "parsed"))
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


@pytest.mark.asyncio
async def test_reingesting_an_unchanged_file_skips_parsing(
    monkeypatch, tmp_path
) -> None:
    """Re-ingest of byte-identical content must not call the model at all."""
    monkeypatch.setenv("RAG_INGEST_ALLOWED_DIRS", str(tmp_path))
    monkeypatch.setattr("app.ingestion.loader._cached_allowed_dirs", None)
    document = tmp_path / "policy.md"
    document.write_text("# Policy\n\n| ID | Value |\n| --- | --- |\n| A-02 | pass |", encoding="utf-8")

    parser = CountingParser()
    pipeline = _pipeline(tmp_path, parser, monkeypatch)

    first = await pipeline.run(str(document), kb_id="default")
    assert first.documents == 1
    assert first.unchanged_doc_ids == []
    calls_after_first = parser.calls
    assert calls_after_first >= 1

    second = await pipeline.run(str(document), kb_id="default")

    # Nothing re-parsed, nothing rewritten, and the skip is reported rather than
    # looking like "the file was not found".
    assert parser.calls == calls_after_first, "unchanged content must not be re-parsed"
    assert second.documents == 0
    assert len(second.unchanged_doc_ids) == 1
    assert second.chunks == 0


@pytest.mark.asyncio
async def test_changed_file_is_reparsed(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RAG_INGEST_ALLOWED_DIRS", str(tmp_path))
    monkeypatch.setattr("app.ingestion.loader._cached_allowed_dirs", None)
    document = tmp_path / "policy.md"
    document.write_text("# Policy\n\nVersion one.\n", encoding="utf-8")

    parser = CountingParser()
    pipeline = _pipeline(tmp_path, parser, monkeypatch)
    await pipeline.run(str(document), kb_id="default")
    calls = parser.calls

    document.write_text("# Policy\n\nVersion two, changed.\n", encoding="utf-8")
    second = await pipeline.run(str(document), kb_id="default")

    assert parser.calls > calls
    assert second.documents == 1
    assert second.unchanged_doc_ids == []


@pytest.mark.asyncio
async def test_force_reparses_an_unchanged_file(monkeypatch, tmp_path) -> None:
    """The explicit rebuild escape hatch still re-parses."""
    monkeypatch.setenv("RAG_INGEST_ALLOWED_DIRS", str(tmp_path))
    monkeypatch.setattr("app.ingestion.loader._cached_allowed_dirs", None)
    document = tmp_path / "policy.md"
    document.write_text("# Policy\n\nText.\n", encoding="utf-8")

    parser = CountingParser()
    pipeline = _pipeline(tmp_path, parser, monkeypatch)
    await pipeline.run(str(document), kb_id="default")
    calls = parser.calls

    forced = await pipeline.run(str(document), kb_id="default", force=True)

    assert parser.calls >= calls  # the cache may serve the text, but work runs
    assert forced.documents == 1
    assert forced.unchanged_doc_ids == []


@pytest.mark.asyncio
async def test_configuration_change_invalidates_the_skip(monkeypatch, tmp_path) -> None:
    """Changing the chunk configuration must not keep serving old chunking."""
    monkeypatch.setenv("RAG_INGEST_ALLOWED_DIRS", str(tmp_path))
    monkeypatch.setattr("app.ingestion.loader._cached_allowed_dirs", None)
    document = tmp_path / "policy.md"
    document.write_text("# Policy\n\nText for chunking.\n", encoding="utf-8")

    parser = CountingParser()
    pipeline = _pipeline(tmp_path, parser, monkeypatch)
    await pipeline.run(str(document), kb_id="default")

    # Rebuild the pipeline with a different chunk size: the fingerprint changes.
    changed = _pipeline(tmp_path, parser, monkeypatch)
    changed.config.settings["chunk"] = {"size": 100, "overlap": 10}
    changed.parse_cache = ParseCache(
        changed.config.parsed_dir,
        build_fingerprint(
            parser_provider=None, parser_model=None, chunk_size=100, chunk_overlap=10
        ),
    )

    response = await changed.run(str(document), kb_id="default")

    assert response.documents == 1
    assert response.unchanged_doc_ids == []


@pytest.mark.asyncio
async def test_artifact_records_the_ingest_fingerprint(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RAG_INGEST_ALLOWED_DIRS", str(tmp_path))
    monkeypatch.setattr("app.ingestion.loader._cached_allowed_dirs", None)
    document = tmp_path / "policy.md"
    document.write_text("# Policy\n\nText.\n", encoding="utf-8")

    pipeline = _pipeline(tmp_path, CountingParser(), monkeypatch)
    await pipeline.run(str(document), kb_id="default")

    artifact = json.loads(next((tmp_path / "parsed").glob("*.json")).read_text(encoding="utf-8"))
    assert artifact["ingest_fingerprint"] == pipeline.parse_cache.fingerprint
    assert _parsed_chunks(tmp_path / "parsed")


@pytest.mark.asyncio
async def test_empty_parse_is_never_cached(tmp_path: Path) -> None:
    """An empty parse must fail, not be cached as a valid result."""
    document = tmp_path / "policy.md"
    document.write_text("# Policy\n\nText.\n", encoding="utf-8")
    cache = _cache(tmp_path)

    with pytest.raises(Exception, match="empty content"):
        await parse_content(document, EmptyDocumentParser(), cache)

    assert not list(cache.root.rglob("*.json"))
