"""Same-content documents inside one ingest batch must not delete each other.

Regression for a data-loss defect: _supersede_same_content excluded only the
document being processed, so two identical files in the SAME batch each saw the
other's freshly committed artifact as a duplicate and removed it. Both artifacts,
wiki pages and graph rows were deleted while the job reported success.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import AppConfig
from app.core.tracing import TracingManager
from app.ingestion.pipeline import IngestionPipeline
from app.tests.test_document_parser import FakeGenerationClient


class _Parser:
    enabled = True

    def supports(self, path: Path) -> bool:  # noqa: ARG002
        return True

    async def parse_file(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")


def _pipeline(tmp_path: Path, monkeypatch) -> IngestionPipeline:
    monkeypatch.setenv("PARSED_OUTPUT_DIR", str(tmp_path / "parsed"))
    monkeypatch.setenv("RAG_INGEST_ALLOWED_DIRS", str(tmp_path))
    monkeypatch.setattr("app.ingestion.loader._cached_allowed_dirs", None)
    config = AppConfig(
        config_dir=tmp_path,
        settings={"timeout": {"document_parser_seconds": 30}},
        models={"model_gateway": {"base_url": "", "api_key": ""}},
        prompts={},
    )
    (tmp_path / "parsed").mkdir(parents=True, exist_ok=True)
    return IngestionPipeline(
        config=config,
        generation_client=FakeGenerationClient(),
        tracing_manager=TracingManager("test-service", ""),
        document_parser=_Parser(),
    )


@pytest.mark.asyncio
async def test_two_identical_files_in_one_batch_survive(monkeypatch, tmp_path) -> None:
    body = "# Policy\n\nSubmit within 15 days."
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "a.md").write_text(body, encoding="utf-8")
    copy_dir = root / "copy"
    copy_dir.mkdir()
    (copy_dir / "a.md").write_text(body, encoding="utf-8")

    pipeline = _pipeline(tmp_path, monkeypatch)
    response = await pipeline.run(str(root), kb_id="default")

    artifacts = list((tmp_path / "parsed").glob("*.json"))
    assert len(artifacts) == 2, (
        f"both same-content documents must survive the batch, found {len(artifacts)}"
    )
    assert response.documents == 2
    # Nothing may be reported as superseded: same-batch siblings are not
    # pre-existing duplicates.
    assert response.superseded_doc_ids == []


@pytest.mark.asyncio
async def test_chunk_config_change_is_not_treated_as_unchanged(monkeypatch, tmp_path) -> None:
    """The ingest fingerprint must cover the REAL chunking knobs.

    It used to read ``settings["chunk"]``, a key that does not exist, so the
    fingerprint was constant: changing chunk sizes then left the artifact
    "unchanged" and the new chunking was silently never applied.
    """
    monkeypatch.setenv("PARSED_OUTPUT_DIR", str(tmp_path / "parsed"))
    monkeypatch.setenv("RAG_INGEST_ALLOWED_DIRS", str(tmp_path))
    monkeypatch.setattr("app.ingestion.loader._cached_allowed_dirs", None)
    document = tmp_path / "policy.md"
    document.write_text("# Policy\n\nSubmit within 15 days.", encoding="utf-8")

    monkeypatch.setenv("RAG_TEXT_CHUNK_MAX_CHARS", "1000")
    first = _pipeline(tmp_path, monkeypatch)
    await first.run(str(document), kb_id="default")
    fingerprint_before = first.parse_cache.fingerprint

    monkeypatch.setenv("RAG_TEXT_CHUNK_MAX_CHARS", "2000")
    second = _pipeline(tmp_path, monkeypatch)
    assert second.parse_cache.fingerprint != fingerprint_before

    response = await second.run(str(document), kb_id="default")
    assert response.documents == 1, "changed chunking must re-ingest, not skip"
    assert response.unchanged_doc_ids == []


class _ExplodingGraphStore:
    """Fails the graph upsert, which happens AFTER the artifact is committed."""

    def __init__(self) -> None:
        self.calls = 0
        self.deleted: list[str] = []

    def upsert_document(self, structured):  # noqa: ANN001, ARG002
        self.calls += 1
        raise RuntimeError("graph store exploded")

    def delete_document(self, *, doc_id: str, kb_id: str) -> None:  # noqa: ARG002
        self.deleted.append(doc_id)

    def collect_orphan_entities(self) -> None:
        return None

    def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_failed_reingest_restores_the_previous_version(monkeypatch, tmp_path) -> None:
    """A failing batch must not destroy the good version it was replacing.

    The old rollback unlinked the artifact of every committed document and deleted
    graph rows for every *prepared* document, so a re-ingest that failed halfway
    destroyed the previously committed document in all three stores.
    """
    monkeypatch.setenv("PARSED_OUTPUT_DIR", str(tmp_path / "parsed"))
    monkeypatch.setenv("RAG_INGEST_ALLOWED_DIRS", str(tmp_path))
    monkeypatch.setattr("app.ingestion.loader._cached_allowed_dirs", None)
    document = tmp_path / "policy.md"
    document.write_text("# Policy\n\nOriginal text.", encoding="utf-8")

    # First ingest succeeds and becomes the version that must survive.
    first = _pipeline(tmp_path, monkeypatch)
    await first.run(str(document), kb_id="default")
    artifacts = list((tmp_path / "parsed").glob("*.json"))
    assert len(artifacts) == 1
    original_bytes = artifacts[0].read_bytes()

    # Second ingest changes the file, then fails on the graph upsert.
    document.write_text("# Policy\n\nReplacement text.", encoding="utf-8")
    graph_store = _ExplodingGraphStore()
    second = _pipeline(tmp_path, monkeypatch)
    second.graph_store = graph_store  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="graph store exploded"):
        await second.run(str(document), kb_id="default")

    restored = list((tmp_path / "parsed").glob("*.json"))
    assert len(restored) == 1, "the artifact must not be deleted by a rollback"
    assert restored[0].read_bytes() == original_bytes, (
        "rollback must restore the previous artifact content, not leave the "
        "half-committed replacement"
    )
    # The pre-existing document must never be deleted from the graph store.
    assert graph_store.deleted == []
    # No staging leftovers.
    assert not list((tmp_path / "parsed").glob(".rollback-*"))
