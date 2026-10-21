"""Content-addressed duplicate detection and removal.

Duplicates are a measurable recall loss: each copy takes a discovery slot and
splits BM25 scores. These tests pin both halves — detection (which copy is
kept, and that legacy artifacts without a stored hash are still detected) and
removal (all three stores are cleaned, or the artifact is at least gone).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.exceptions import StoreError
from app.ingestion.dedup import (
    content_fingerprint,
    find_duplicate_documents,
    find_duplicates_for_content,
    iter_fingerprints,
    remove_document_everywhere,
)


def _write_artifact(
    parsed_dir: Path,
    doc_id: str,
    *,
    content: str,
    kb_id: str = "default",
    source_path: str = "doc.md",
    source_content_hash: str | None = None,
    mtime: float | None = None,
) -> Path:
    metadata: dict[str, object] = {"kb_id": kb_id}
    if source_content_hash:
        metadata["source_content_hash"] = source_content_hash
    payload = {
        "document": {
            "doc_id": doc_id,
            "source_path": source_path,
            "title": doc_id,
            "content": content,
            "metadata": metadata,
        },
        "chunks": [],
    }
    path = parsed_dir / f"{doc_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    if mtime is not None:
        import os

        os.utime(path, (mtime, mtime))
    return path


# --------------------------------------------------------------------------- #
# Detection                                                                   #
# --------------------------------------------------------------------------- #

def test_identical_content_in_one_kb_is_grouped(tmp_path: Path) -> None:
    _write_artifact(tmp_path, "a", content="same text", source_path="one.md")
    _write_artifact(tmp_path, "b", content="same text", source_path="two.md")

    groups = find_duplicate_documents(iter_fingerprints(tmp_path))

    assert len(groups) == 1
    assert {member.doc_id for member in groups[0].redundant} == {"a"}
    assert len(groups[0].redundant) == 1


def test_same_content_in_different_kbs_is_not_a_duplicate(tmp_path: Path) -> None:
    _write_artifact(tmp_path, "a", content="same text", kb_id="default")
    _write_artifact(tmp_path, "b", content="same text", kb_id="finance")

    assert find_duplicate_documents(iter_fingerprints(tmp_path)) == []


def test_different_content_is_not_a_duplicate(tmp_path: Path) -> None:
    _write_artifact(tmp_path, "a", content="alpha")
    _write_artifact(tmp_path, "b", content="beta")

    assert find_duplicate_documents(iter_fingerprints(tmp_path)) == []


def test_current_format_and_stable_id_win_over_legacy_copy(tmp_path: Path) -> None:
    # Legacy artifact: no stored content hash, opaque doc_id.
    _write_artifact(tmp_path, "31ae51ae-legacy", content="same text", mtime=1_000)
    # Current artifact: has provenance and a stable id.
    _write_artifact(
        tmp_path,
        "doc-abc123",
        content="same text",
        source_content_hash="deadbeef",
        mtime=500,
    )

    groups = find_duplicate_documents(iter_fingerprints(tmp_path))

    assert len(groups) == 1
    assert groups[0].keep.doc_id == "doc-abc123"
    assert [member.doc_id for member in groups[0].redundant] == ["31ae51ae-legacy"]


def test_duplicates_are_detected_without_a_stored_content_hash(tmp_path: Path) -> None:
    # Legacy artifacts predate source_content_hash; keying on it would find no
    # duplicates in exactly the corpora that need cleaning.
    _write_artifact(tmp_path, "a", content="same text")
    _write_artifact(tmp_path, "b", content="same text")

    fingerprints = iter_fingerprints(tmp_path)
    assert all(fingerprint.source_content_hash is None for fingerprint in fingerprints)
    assert len(find_duplicate_documents(fingerprints)) == 1


def test_artifact_without_text_is_ignored(tmp_path: Path) -> None:
    # Media-only artifacts carry no document text: they cannot be compared by
    # content and must not be grouped together as duplicates of each other.
    _write_artifact(tmp_path, "m1", content="")
    _write_artifact(tmp_path, "m2", content="")

    assert iter_fingerprints(tmp_path) == []


def test_corrupt_artifact_raises(tmp_path: Path) -> None:
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(StoreError):
        iter_fingerprints(tmp_path)


def test_find_duplicates_for_content_excludes_the_incoming_document(tmp_path: Path) -> None:
    _write_artifact(tmp_path, "old", content="body text", source_path="old/path.md")
    _write_artifact(tmp_path, "new", content="body text", source_path="new/path.md")

    matches = find_duplicates_for_content(
        tmp_path, kb_id="default", content="body text", exclude_doc_id="new"
    )

    assert [match.doc_id for match in matches] == ["old"]


def test_find_duplicates_for_content_scopes_by_kb(tmp_path: Path) -> None:
    _write_artifact(tmp_path, "other", content="body text", kb_id="finance")

    assert (
        find_duplicates_for_content(tmp_path, kb_id="default", content="body text")
        == []
    )


def test_stale_identity_detected_across_path_roots(tmp_path: Path) -> None:
    # A bare local script recorded a repo-relative path; the container recorded
    # an absolute one. Same file, two identities, different text (parser
    # version), so content matching cannot see it.
    _write_artifact(
        tmp_path,
        "31ae51ae-legacy",
        content="old degenerate parse",
        source_path="/workspace/data/raw/handbook.md",
        mtime=1_000,
    )
    _write_artifact(
        tmp_path,
        "doc-abc",
        content="current parse",
        source_path="data/raw/handbook.md",
        source_content_hash="deadbeef",
        mtime=500,
    )

    groups = find_duplicate_documents(iter_fingerprints(tmp_path))
    stale = [group for group in groups if group.reason == "stale_identity"]

    assert len(stale) == 1
    assert stale[0].keep.doc_id == "doc-abc"
    assert [member.doc_id for member in stale[0].redundant] == ["31ae51ae-legacy"]


def test_different_directories_with_same_basename_are_not_stale(tmp_path: Path) -> None:
    # reports/spec.pdf and contracts/spec.pdf are different documents; a single
    # shared basename must not be enough to declare one stale.
    _write_artifact(
        tmp_path, "legacy", content="a", source_path="/workspace/reports/spec.pdf"
    )
    _write_artifact(
        tmp_path,
        "doc-x",
        content="b",
        source_path="contracts/spec.pdf",
        source_content_hash="deadbeef",
    )

    groups = find_duplicate_documents(iter_fingerprints(tmp_path))
    assert [group for group in groups if group.reason == "stale_identity"] == []


def test_two_legacy_artifacts_are_not_treated_as_stale(tmp_path: Path) -> None:
    # Without a current-format member there is no upgrade to resolve; legacy
    # corpora must not be pruned by a path heuristic alone.
    _write_artifact(
        tmp_path, "legacy-a", content="a", source_path="/workspace/data/raw/x.md"
    )
    _write_artifact(
        tmp_path, "legacy-b", content="b", source_path="data/raw/x.md"
    )

    groups = find_duplicate_documents(iter_fingerprints(tmp_path))
    assert [group for group in groups if group.reason == "stale_identity"] == []


def test_content_fingerprint_is_stable() -> None:
    assert content_fingerprint("abc") == content_fingerprint("abc")
    assert content_fingerprint("abc") != content_fingerprint("abd")


# --------------------------------------------------------------------------- #
# Removal                                                                     #
# --------------------------------------------------------------------------- #

def test_removal_clears_artifact_and_attachments(tmp_path: Path) -> None:
    artifact = _write_artifact(tmp_path, "dup", content="text")
    attachments = tmp_path / "attachments" / "dup"
    attachments.mkdir(parents=True)
    (attachments / "page-1.pdf").write_text("payload", encoding="utf-8")

    touched = remove_document_everywhere(doc_id="dup", kb_id="default", parsed_dir=tmp_path)

    assert not artifact.exists()
    assert not attachments.exists()
    assert set(touched) == {"artifact", "attachments"}


def test_removal_reaches_graph_and_wiki(tmp_path: Path) -> None:
    _write_artifact(tmp_path, "dup", content="text")

    class FakeGraph:
        def __init__(self) -> None:
            self.deleted: list[tuple[str, str]] = []

        def delete_document(self, *, doc_id: str, kb_id: str) -> None:
            self.deleted.append((doc_id, kb_id))

    class FakeWiki:
        def __init__(self) -> None:
            self.removed: list[str] = []

        def remove_document(self, doc_id: str) -> None:
            self.removed.append(doc_id)

    graph, wiki = FakeGraph(), FakeWiki()
    touched = remove_document_everywhere(
        doc_id="dup",
        kb_id="default",
        parsed_dir=tmp_path,
        graph_store=graph,  # type: ignore[arg-type]
        wiki_compiler=wiki,  # type: ignore[arg-type]
    )

    assert graph.deleted == [("dup", "default")]
    assert wiki.removed == ["dup"]
    assert set(touched) == {"artifact", "graph", "wiki"}


def test_removal_of_missing_document_reports_nothing(tmp_path: Path) -> None:
    assert remove_document_everywhere(doc_id="ghost", kb_id="default", parsed_dir=tmp_path) == []


# --------------------------------------------------------------------------- #
# Ingest path: a file arriving under a new path supersedes the old copy       #
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_ingest_supersedes_same_content_at_a_new_path(
    monkeypatch, tmp_path
) -> None:
    """Uploading a file that already exists elsewhere must not duplicate it.

    Two documents with identical content compete for the same discovery slots;
    the copy is removed and the removal is reported on the response.
    """
    from app.core.config import AppConfig
    from app.core.tracing import TracingManager
    from app.ingestion.pipeline import IngestionPipeline

    source_root = tmp_path / "raw"
    source_root.mkdir()
    parsed_dir = tmp_path / "parsed"
    monkeypatch.setenv("RAG_INGEST_ALLOWED_DIRS", str(source_root))
    monkeypatch.setenv("PARSED_OUTPUT_DIR", str(parsed_dir))
    monkeypatch.setattr("app.ingestion.loader._cached_allowed_dirs", None)

    body = "# Policy\n\nSubmit the form within 15 days.\n"
    first = source_root / "policy.md"
    first.write_text(body, encoding="utf-8")
    second = source_root / "policy_copy.md"
    second.write_text(body, encoding="utf-8")

    class _Generation:
        async def generate(self, messages):  # noqa: ANN001, ARG002
            return {"content": '{"entities": [], "relations": []}'}

    config = AppConfig(
        config_dir=tmp_path,
        settings={
            "chunk": {"size": 200, "overlap": 20},
            "timeout": {"document_parser_seconds": 30},
        },
        models={"model_gateway": {"base_url": "", "api_key": ""}},
        prompts={},
    )
    pipeline = IngestionPipeline(
        config=config,
        generation_client=_Generation(),
        tracing_manager=TracingManager("test-service", ""),
        document_parser=None,
    )

    first_response = await pipeline.run(str(first), kb_id="default")
    assert first_response.superseded_doc_ids == []

    second_response = await pipeline.run(str(second), kb_id="default")

    # The copy was superseded exactly once, and only one document remains.
    assert len(second_response.superseded_doc_ids) == 1
    artifacts = sorted(parsed_dir.glob("*.json"))
    assert len(artifacts) == 1
    # The superseded doc is the *other* document, and the surviving artifact is
    # the newly ingested path.
    remaining = json.loads(artifacts[0].read_text(encoding="utf-8"))["document"]
    assert remaining["source_path"].endswith("policy_copy.md")
    assert artifacts[0].stem != second_response.superseded_doc_ids[0]


@pytest.mark.asyncio
async def test_ingest_does_not_supersede_different_content(monkeypatch, tmp_path) -> None:
    from app.core.config import AppConfig
    from app.core.tracing import TracingManager
    from app.ingestion.pipeline import IngestionPipeline

    source_root = tmp_path / "raw"
    source_root.mkdir()
    parsed_dir = tmp_path / "parsed"
    monkeypatch.setenv("RAG_INGEST_ALLOWED_DIRS", str(source_root))
    monkeypatch.setenv("PARSED_OUTPUT_DIR", str(parsed_dir))
    monkeypatch.setattr("app.ingestion.loader._cached_allowed_dirs", None)

    (source_root / "a.md").write_text("# A\n\nalpha body\n", encoding="utf-8")
    (source_root / "b.md").write_text("# B\n\nbeta body\n", encoding="utf-8")

    class _Generation:
        async def generate(self, messages):  # noqa: ANN001, ARG002
            return {"content": '{"entities": [], "relations": []}'}

    config = AppConfig(
        config_dir=tmp_path,
        settings={
            "chunk": {"size": 200, "overlap": 20},
            "timeout": {"document_parser_seconds": 30},
        },
        models={"model_gateway": {"base_url": "", "api_key": ""}},
        prompts={},
    )
    pipeline = IngestionPipeline(
        config=config,
        generation_client=_Generation(),
        tracing_manager=TracingManager("test-service", ""),
        document_parser=None,
    )

    await pipeline.run(str(source_root / "a.md"), kb_id="default")
    second = await pipeline.run(str(source_root / "b.md"), kb_id="default")

    assert second.superseded_doc_ids == []
    assert len(sorted(parsed_dir.glob("*.json"))) == 2
