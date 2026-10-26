"""Discovery is document-level: one statute takes one candidate slot.

Why this module exists. The wiki index holds one page per article of a long document
plus one page for the document itself, which is what makes a single clause findable.
The read plan, however, chooses DOCUMENTS to deep-read - and it was handed a page
ranking, so a statute that matched on six of its articles consumed six of the twelve
candidate slots (render: the same title six times, with different one-article excerpts)
and pushed the document that actually answered the question out of the top twelve.

Measured on 270 blind title queries over 595 real statutes: collapsing the ranking to
one page per document moved hit@4 from 0.7407 to 0.7741 and hit@12 from 0.8519 to
0.8963, and cut the never-retrieved count from 40 to 28. The page ranking itself was
unchanged, so these tests pin the collapse, not the scoring.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.tests.test_agentic_discovery import (
    ReadingPlanClient,
    _chunk,
    _config,
    _discovery,
    _doc,
    _write_artifact,
)
from app.wiki.compiler import WikiCompiler
from app.wiki.search import WikiSearcher


def _build_article_corpus(tmp_path: Path) -> tuple[WikiSearcher, Path]:
    """One long statute (many article pages), plus decoys that match a few terms."""
    wiki_dir = tmp_path / "wiki"
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    compiler = WikiCompiler(wiki_dir)

    articles = [
        ("第一条", "为保障供热安全，制定本条例。"),
        ("第二条", "本市行政区域内的供热用热活动，适用本条例。"),
        ("第三条", "供热期为每年十一月一日至次年三月三十一日。"),
        ("第四条", "供热单位应当按照技术标准维护供热设施。"),
        ("第五条", "用户应当按照规定交纳供热费。"),
        ("第六条", "供热设施发生故障的，供热单位应当及时抢修。"),
    ]
    statute = _doc(
        "doc-heat",
        "heating regulations",
        "v1",
        "2025-01-01",
        "# Heating Regulations\n\n" + "".join(text for _label, text in articles),
    )
    chunks = [_chunk("doc-heat", label, text) for label, text in articles]
    compiler.upsert_document(statute, chunks)
    compiler.write_article_pages(statute, articles)
    _write_artifact(parsed_dir, "doc-heat", statute, chunks)

    # Decoys: they mention 供热 or 供热期 once each, so on a page-level ranking they
    # compete with the statute's own six article pages.
    decoys = [
        ("doc-decoy-a", "城市管理条例", "城市管理活动中，供热设施应当定期检查。"),
        ("doc-decoy-b", "建筑市场条例", "建筑工程的供热系统应当符合标准。"),
        ("doc-decoy-c", "燃气安全条例", "燃气管道的维护由经营单位负责。"),
    ]
    for doc_id, title, text in decoys:
        document = _doc(doc_id, title, "v1", "2025-01-01", f"# {title}\n\n{text}")
        document_chunk = _chunk(doc_id, "总则", text)
        compiler.upsert_document(document, [document_chunk])
        _write_artifact(parsed_dir, doc_id, document, [document_chunk])

    return WikiSearcher(wiki_dir), parsed_dir


def _search_hits(searcher: WikiSearcher, query: str, **kwargs) -> list:
    return searcher.search(query, top_k=6, kb_id="default", **kwargs)


def test_document_granularity_reports_each_document_once(tmp_path: Path) -> None:
    searcher, _parsed = _build_article_corpus(tmp_path)

    page_hits = _search_hits(searcher, "供热期 供热设施")
    document_hits = _search_hits(searcher, "供热期 供热设施", granularity="document")

    page_doc_ids = [hit.chunk.doc_id for hit in page_hits]
    assert page_doc_ids.count("doc-heat") > 1, (
        "fixture must give the long statute several matching pages, otherwise this "
        "test cannot detect the duplication it exists for"
    )
    document_doc_ids = [hit.chunk.doc_id for hit in document_hits]
    assert document_doc_ids.count("doc-heat") == 1
    assert len(document_doc_ids) == len(set(document_doc_ids))
    # Collapse keeps the best page, so the winning page is the same one the page
    # ranking put first - the collapse must not reorder anything else.
    assert document_doc_ids[0] == page_doc_ids[0]


def test_document_granularity_promotes_the_answer_into_the_candidate_set(
    tmp_path: Path,
) -> None:
    """The regression this change fixes, on a corpus small enough to reason about."""
    searcher, _parsed = _build_article_corpus(tmp_path)

    page_doc_ids = [hit.chunk.doc_id for hit in _search_hits(searcher, "供热期 供热设施")]
    document_doc_ids = [
        hit.chunk.doc_id
        for hit in _search_hits(searcher, "供热期 供热设施", granularity="document")
    ]

    # The statute answers the question and must be in the candidate set under both
    # rankings; what changes is how many slots its own pages take.
    assert "doc-heat" in page_doc_ids
    assert "doc-heat" in document_doc_ids
    assert len(set(page_doc_ids)) < len(page_doc_ids)


def test_article_hit_reports_the_parent_documents_provenance(tmp_path: Path) -> None:
    """An article page must not look like an unversioned document.

    The article page's own frontmatter carries its label, not ``source_key`` or
    ``is_latest_version``. Reporting those as absent would let an article hit slip past
    the deterministic version filter - a stale document would be read as if it were the
    current one, or a current one would be filtered as unrelated.
    """
    searcher, _parsed = _build_article_corpus(tmp_path)

    article_hits = [
        hit
        for hit in searcher.search("供热期", top_k=12, kb_id="default")
        if (hit.chunk.metadata or {}).get("wiki_kind") == "article"
    ]
    # The page-level ranking is what surfaces article pages; the document-level one
    # reports one page per document.
    assert article_hits, "fixture must produce at least one article hit"
    for hit in article_hits:
        metadata = hit.chunk.metadata or {}
        assert metadata["source_key"] == "heating regulations"
        assert metadata["is_latest_version"] is True
        assert metadata["effective_date"] == "2025-01-01"
        # The label is still the page's own, so the plan can name the article it saw.
        assert metadata["wiki_kind"] == "article"
        assert metadata["wiki_page_path"]


def test_article_hit_still_points_at_a_readable_document(tmp_path: Path) -> None:
    searcher, _parsed = _build_article_corpus(tmp_path)

    hits = searcher.search("供热期", top_k=12, kb_id="default")
    readable = [
        hit
        for hit in hits
        if (hit.chunk.metadata or {}).get("wiki_kind") in {"source", "article"}
    ]
    assert readable
    # Every readable hit resolves to a document the deep read can open, and the
    # statute's article hits carry the statute's doc_id, not a page-local id.
    statute_hits = [hit for hit in readable if "doc-heat" in hit.chunk.doc_id]
    assert statute_hits
    assert all(hit.chunk.doc_id == "doc-heat" for hit in statute_hits)


def test_plan_receives_one_manifest_per_document(tmp_path: Path) -> None:
    """End to end: the read plan's manifest must not repeat a document."""
    searcher, parsed_dir = _build_article_corpus(tmp_path)
    config = _config(tmp_path)
    config.parsed_dir = parsed_dir
    client = ReadingPlanClient()
    discovery = _discovery(config, searcher, client)

    staged = asyncio.run(discovery.discovery_candidates("供热期 供热设施", "default"))

    doc_ids = [hit.chunk.doc_id for hit in staged.candidates]
    assert len(doc_ids) == len(set(doc_ids))
    assert "doc-heat" in doc_ids


def test_page_granularity_remains_available(tmp_path: Path) -> None:
    """The page ranking is the default and is unchanged: article-level matching is
    the reason the wiki indexes articles, and a caller that wants "which passage
    matched" (the debug/search API) still gets it."""
    searcher, _parsed = _build_article_corpus(tmp_path)

    hits = searcher.search("供热期", top_k=12, kb_id="default")
    kinds = {(hit.chunk.metadata or {}).get("wiki_kind") for hit in hits}
    assert "article" in kinds
