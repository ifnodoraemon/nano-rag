import json

from app.schemas.chunk import Chunk
from app.schemas.document import Document
from app.wiki.compiler import WikiCompiler
from app.wiki.search import WikiSearcher


def test_wiki_search_filters_pages_by_kb(tmp_path) -> None:
    wiki_compiler = WikiCompiler(tmp_path / "wiki")
    wiki_compiler.upsert_document(
        Document(
            doc_id="default-doc",
            source_path="uploads/default/policy.md",
            title="Default Policy",
            content="# Policy\n\nDefault vacation rules.",
            metadata={"kb_id": "default"},
        ),
        [
            Chunk(
                chunk_id="default:0",
                doc_id="default-doc",
                chunk_index=0,
                text="Default vacation rules.",
                source_path="uploads/default/policy.md",
                metadata={"kb_id": "default"},
            )
        ],
    )
    wiki_compiler.upsert_document(
        Document(
            doc_id="other-doc",
            source_path="uploads/other/policy.md",
            title="Other Policy",
            content="# Policy\n\nOther vacation rules.",
            metadata={"kb_id": "other"},
        ),
        [
            Chunk(
                chunk_id="other:0",
                doc_id="other-doc",
                chunk_index=0,
                text="Other vacation rules.",
                source_path="uploads/other/policy.md",
                metadata={"kb_id": "other"},
            )
        ],
    )
    wiki_searcher = WikiSearcher(tmp_path / "wiki")

    default_hits = wiki_searcher.search("vacation", top_k=10, kb_id="default")
    other_hits = wiki_searcher.search("vacation", top_k=10, kb_id="other")

    assert default_hits
    assert other_hits
    assert all(hit.chunk.metadata.get("kb_id") == "default" for hit in default_hits)
    assert all(hit.chunk.metadata.get("kb_id") == "other" for hit in other_hits)




def _wiki_doc(doc_id: str, source_key: str, version: str, effective_date: str) -> Document:
    return Document(
        doc_id=doc_id,
        source_path=f"uploads/default/{doc_id}.md",
        title=f"{source_key.title()} {version}",
        content=f"# {source_key.title()}\n\n{version} rules apply.",
        metadata={
            "kb_id": "default",
            "source_key": source_key,
            "doc_type": "document",
            "effective_date": effective_date,
            "version": version,
            "source_content_hash": f"hash-{doc_id}",
        },
    )


def _wiki_chunk(doc_id: str) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}:0",
        doc_id=doc_id,
        chunk_index=0,
        text=f"{doc_id} body",
        source_path=f"uploads/default/{doc_id}.md",
        title=doc_id,
        metadata={"kb_id": "default"},
    )


def test_wiki_source_page_frontmatter_carries_version_fields(tmp_path) -> None:
    compiler = WikiCompiler(tmp_path / "wiki")
    compiler.upsert_document(_wiki_doc("doc-a", "travel policy", "v1", None), [_wiki_chunk("doc-a")])

    metadata, _ = WikiCompiler.read_frontmatter(tmp_path / "wiki" / "sources" / "doc-a.md")
    assert metadata["content_hash"] == "hash-doc-a"
    assert metadata["is_latest_version"] is True
    assert metadata["superseded_by"] is None
    assert metadata["version"] == "v1"
    assert metadata["source_key"] == "travel policy"


def test_wiki_version_chain_marks_superseded_pages(tmp_path) -> None:
    compiler = WikiCompiler(tmp_path / "wiki")
    compiler.upsert_document(_wiki_doc("doc-v1", "employee manual", "v1", "2024-01-01"), [_wiki_chunk("doc-v1")])
    compiler.upsert_document(_wiki_doc("doc-v2", "employee manual", "v2", "2025-06-01"), [_wiki_chunk("doc-v2")])
    compiler.upsert_document(_wiki_doc("doc-solo", "travel policy", "v1", None), [_wiki_chunk("doc-solo")])

    old, _ = WikiCompiler.read_frontmatter(tmp_path / "wiki" / "sources" / "doc-v1.md")
    new, _ = WikiCompiler.read_frontmatter(tmp_path / "wiki" / "sources" / "doc-v2.md")
    solo, _ = WikiCompiler.read_frontmatter(tmp_path / "wiki" / "sources" / "doc-solo.md")

    assert new["is_latest_version"] is True
    assert new["superseded_by"] is None
    assert old["is_latest_version"] is False
    assert old["superseded_by"] == "doc-v2"
    assert solo["is_latest_version"] is True
    assert solo["superseded_by"] is None


def test_wiki_version_chain_prefers_date_then_version(tmp_path) -> None:
    compiler = WikiCompiler(tmp_path / "wiki")
    # Two docs, same source_key, same version string, different dates.
    compiler.upsert_document(_wiki_doc("doc-d1", "hr policy", "v1", "2023-01-01"), [_wiki_chunk("doc-d1")])
    compiler.upsert_document(_wiki_doc("doc-d2", "hr policy", "v1", "2024-05-05"), [_wiki_chunk("doc-d2")])

    d1, _ = WikiCompiler.read_frontmatter(tmp_path / "wiki" / "sources" / "doc-d1.md")
    d2, _ = WikiCompiler.read_frontmatter(tmp_path / "wiki" / "sources" / "doc-d2.md")
    assert d2["is_latest_version"] is True
    assert d1["is_latest_version"] is False
    assert d1["superseded_by"] == "doc-d2"


def test_wiki_version_chain_recomputed_after_remove(tmp_path) -> None:
    compiler = WikiCompiler(tmp_path / "wiki")
    compiler.upsert_document(_wiki_doc("doc-v1", "employee manual", "v1", "2024-01-01"), [_wiki_chunk("doc-v1")])
    compiler.upsert_document(_wiki_doc("doc-v2", "employee manual", "v2", "2025-06-01"), [_wiki_chunk("doc-v2")])

    compiler.remove_document("doc-v2")

    d1, _ = WikiCompiler.read_frontmatter(tmp_path / "wiki" / "sources" / "doc-v1.md")
    assert d1["is_latest_version"] is True
    assert d1["superseded_by"] is None


def test_wiki_search_hit_metadata_includes_version_fields(tmp_path) -> None:
    compiler = WikiCompiler(tmp_path / "wiki")
    compiler.upsert_document(_wiki_doc("doc-v1", "employee manual", "v1", "2024-01-01"), [_wiki_chunk("doc-v1")])
    compiler.upsert_document(_wiki_doc("doc-v2", "employee manual", "v2", "2025-06-01"), [_wiki_chunk("doc-v2")])
    searcher = WikiSearcher(tmp_path / "wiki")

    hits = searcher.search("employee manual rules", top_k=10, kb_id="default")
    by_id = {hit.chunk.doc_id: hit for hit in hits if hit.chunk.doc_id in {"doc-v1", "doc-v2"}}

    assert by_id["doc-v2"].chunk.metadata["is_latest_version"] is True
    assert by_id["doc-v2"].chunk.metadata["superseded_by"] is None
    assert by_id["doc-v1"].chunk.metadata["is_latest_version"] is False
    assert by_id["doc-v1"].chunk.metadata["superseded_by"] == "doc-v2"


def test_wiki_bootstrap_rebuilds_from_parsed_artifacts(tmp_path) -> None:
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    for doc_id, version, effective_date in [
        ("doc-v1", "v1", "2024-01-01"),
        ("doc-v2", "v2", "2025-06-01"),
    ]:
        document = _wiki_doc(doc_id, "employee manual", version, effective_date)
        (parsed_dir / f"{doc_id}.json").write_text(
            json.dumps(
                {
                    "document": document.model_dump(),
                    "chunks": [_wiki_chunk(doc_id).model_dump()],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    # A fresh wiki dir (as after a container rebuild) rebuilds from artifacts.
    wiki_dir = tmp_path / "wiki"
    compiler = WikiCompiler(wiki_dir)
    count = compiler.bootstrap_from_parsed_dir(parsed_dir)

    assert count == 2
    assert (wiki_dir / "sources" / "doc-v1.md").exists()
    assert (wiki_dir / "sources" / "doc-v2.md").exists()

    old, _ = WikiCompiler.read_frontmatter(wiki_dir / "sources" / "doc-v1.md")
    new, _ = WikiCompiler.read_frontmatter(wiki_dir / "sources" / "doc-v2.md")
    assert new["is_latest_version"] is True
    assert old["is_latest_version"] is False
    assert old["superseded_by"] == "doc-v2"

    # The searcher re-indexes the rebuilt pages and can find them.
    searcher = WikiSearcher(wiki_dir)
    hits = searcher.search("employee manual rules", top_k=10, kb_id="default")
    source_hits = {
        hit.chunk.doc_id for hit in hits if hit.chunk.metadata.get("wiki_kind") == "source"
    }
    assert source_hits == {"doc-v1", "doc-v2"}

    # Bootstrap is idempotent and does not append to the ingest log.
    assert (wiki_dir / "log.md").read_text(encoding="utf-8").count("ingest") == 0
    assert compiler.bootstrap_from_parsed_dir(parsed_dir) == 2


def test_wiki_searcher_reindexes_pages_written_by_another_process(tmp_path) -> None:
    """Cross-process staleness: in the standard runtime the Celery ingest
    worker writes wiki pages while the app serves search. The worker's
    refresh() cannot reach the app's in-memory index, so the app's searcher
    must self-heal on the next search instead of staying blind to every
    document ingested after it started."""
    wiki_dir = tmp_path / "wiki"
    # App process: searcher constructed at startup, before any page exists.
    searcher = WikiSearcher(wiki_dir)
    assert searcher.search("travel policy", top_k=10, kb_id="default") == []

    # Worker process (simulated by a second compiler instance): writes pages
    # to the shared wiki directory. It refreshes its OWN searcher, not the
    # app's — the app's searcher is never told about the new pages.
    worker_compiler = WikiCompiler(wiki_dir)
    worker_compiler.upsert_document(
        _wiki_doc("travel-doc", "travel policy", "v1", "2025-01-01"),
        [_wiki_chunk("travel-doc")],
    )

    # Next search in the app process must pick up the newly written page.
    hits = searcher.search("travel policy", top_k=10, kb_id="default")
    source_hits = {
        hit.chunk.doc_id for hit in hits if hit.chunk.metadata.get("wiki_kind") == "source"
    }
    assert source_hits == {"travel-doc"}

    # A removal written out-of-band is also picked up (not just additions).
    # Only the source page is removed here, so topic/index pages that the
    # compiler also wrote may still match — but the source hit must vanish.
    (wiki_dir / "sources" / "travel-doc.md").unlink()
    hits_after_remove = searcher.search("travel policy", top_k=10, kb_id="default")
    source_hits_after = {
        hit.chunk.doc_id for hit in hits_after_remove if hit.chunk.metadata.get("wiki_kind") == "source"
    }
    assert source_hits_after == set()


def test_wiki_bootstrap_fails_loud_on_malformed_artifacts(tmp_path) -> None:
    # No silent skipping: corrupt parsed artifacts are a data-integrity
    # failure and must raise at bootstrap instead of being quietly dropped.
    import pytest

    from app.core.exceptions import ParsingError

    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    (parsed_dir / "good.json").write_text(
        json.dumps(
            {
                "document": _wiki_doc("doc-a", "travel policy", "v1", None).model_dump(),
                "chunks": [_wiki_chunk("doc-a").model_dump()],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (parsed_dir / "broken.json").write_text("{not valid json", encoding="utf-8")
    (parsed_dir / "missing-doc.json").write_text(json.dumps({"chunks": []}), encoding="utf-8")

    wiki_dir = tmp_path / "wiki"
    compiler = WikiCompiler(wiki_dir)
    with pytest.raises(ParsingError, match="corrupt parsed artifacts"):
        compiler.bootstrap_from_parsed_dir(parsed_dir)

    # The failure was raised before any page was written.
    assert not (wiki_dir / "sources" / "doc-a.md").exists()


def test_wiki_searcher_per_kb_quota_raises(monkeypatch, tmp_path) -> None:
    # A per-KB page ceiling must fail visibly instead of silently dropping
    # pages from the index.
    import pytest as _pytest

    from app.core.exceptions import RetrievalError

    monkeypatch.setenv("RAG_WIKI_MAX_PAGES_PER_KB", "2")
    compiler = WikiCompiler(tmp_path / "wiki")
    for doc_id in ("doc-a", "doc-b"):
        compiler.upsert_document(_wiki_doc(doc_id, f"policy {doc_id}", "v1", None), [_wiki_chunk(doc_id)])

    from app.wiki.search import WikiSearcher as _Searcher

    searcher = _Searcher(tmp_path / "wiki")
    assert searcher.stats()["source_pages"] == 2

    compiler.upsert_document(_wiki_doc("doc-c", "policy c", "v1", None), [_wiki_chunk("doc-c")])
    with _pytest.raises(RetrievalError, match="quota"):
        searcher.search("policy", top_k=10, kb_id="default")


def test_wiki_searcher_incremental_sync_avoids_full_reindex(monkeypatch, tmp_path) -> None:
    # Only changed/new pages may be re-indexed on sync; unchanged pages must
    # not be re-read (the old any-change-means-full-rebuild behavior caused a
    # query-side re-index storm during ingest).
    wiki_dir = tmp_path / "wiki"
    compiler = WikiCompiler(wiki_dir)
    compiler.upsert_document(_wiki_doc("doc-a", "policy a", "v1", None), [_wiki_chunk("doc-a")])
    searcher = WikiSearcher(wiki_dir)
    assert searcher.search("policy", top_k=10, kb_id="default")

    calls: list[str] = []
    original = searcher._index_page

    def counting_index_page(path):
        calls.append(path.name)
        return original(path)

    monkeypatch.setattr(searcher, "_index_page", counting_index_page)
    # No changes on disk: sync must not re-index anything.
    assert searcher.search("policy", top_k=10, kb_id="default")
    assert calls == []

    # One new page: exactly that page is indexed.
    compiler.upsert_document(_wiki_doc("doc-b", "policy b", "v1", None), [_wiki_chunk("doc-b")])
    hits = searcher.search("policy b", top_k=10, kb_id="default")
    assert "doc-b" in {
        hit.chunk.doc_id for hit in hits if hit.chunk.metadata.get("wiki_kind") == "source"
    }
    assert "doc-b.md" in calls
    assert "doc-a.md" not in calls
