import pytest

from app.core.exceptions import ParsingError, RetrievalError
from app.schemas.chunk import Chunk
from app.schemas.document import Document
from app.wiki.compiler import WikiCompiler


def test_wiki_compiler_writes_source_page_index_and_log(tmp_path) -> None:
    compiler = WikiCompiler(tmp_path / "wiki")
    document = Document(
        doc_id="doc-1",
        source_path="data/raw/handbook.md",
        title="Employee Handbook",
        content=(
            "# Leave Policy\n\n"
            "Employees can carry over PTO into the next year subject to manager approval.\n\n"
            "## Expense Rules\n\n"
            "Expense reimbursements should be filed within thirty days."
        ),
        metadata={"kb_id": "default"},
    )
    chunks = [
        Chunk(
            chunk_id="doc-1:0",
            doc_id="doc-1",
            chunk_index=0,
            text="Employees can carry over PTO into the next year subject to manager approval.",
            source_path="data/raw/handbook.md",
            title="Employee Handbook",
            metadata={"kb_id": "default"},
        )
    ]

    page_path = compiler.upsert_document(document, chunks)

    source_page = page_path.read_text(encoding="utf-8")
    index_page = (tmp_path / "wiki" / "index.md").read_text(encoding="utf-8")
    topic_page = (tmp_path / "wiki" / "topics" / "default--leave-policy.md").read_text(
        encoding="utf-8"
    )
    scoped_index_page = (tmp_path / "wiki" / "indexes" / "default.md").read_text(
        encoding="utf-8"
    )
    log_page = (tmp_path / "wiki" / "log.md").read_text(encoding="utf-8")
    schema_page = (tmp_path / "wiki" / "SCHEMA.md").read_text(encoding="utf-8")

    assert "# Employee Handbook" in source_page
    assert "# Leave Policy" in topic_page
    assert "## Status" in topic_page
    assert "[Employee Handbook](../sources/doc-1.md)" in topic_page
    assert "Leave Policy" in source_page
    assert "`doc-1:0`" in source_page
    assert "[Employee Handbook](sources/doc-1.md)" in index_page
    assert "[Leave Policy](topics/default--leave-policy.md)" in index_page
    assert "[Employee Handbook](../sources/doc-1.md)" in scoped_index_page
    # The scoped index page carries the real kb_id in frontmatter so the
    # searcher can attribute it without reversing the scope id.
    assert "kb_id: default" in scoped_index_page
    assert "ingest | Employee Handbook" in log_page
    assert "compiled knowledge layer" in schema_page


def test_wiki_compiler_topic_page_aggregates_facts(tmp_path) -> None:
    compiler = WikiCompiler(tmp_path / "wiki")
    compiler.upsert_document(
        Document(
            doc_id="doc-1",
            source_path="data/raw/policy_a.md",
            title="Policy A",
            content="# Leave Policy\n\nPTO carryover is allowed up to 5 days.",
            metadata={"kb_id": "default"},
        ),
        [
            Chunk(
                chunk_id="doc-1:0",
                doc_id="doc-1",
                chunk_index=0,
                text="PTO carryover is allowed up to 5 days.",
                source_path="data/raw/policy_a.md",
                title="Policy A",
                metadata={"kb_id": "default"},
            )
        ],
    )
    compiler.upsert_document(
        Document(
            doc_id="doc-2",
            source_path="data/raw/policy_b.md",
            title="Policy B",
            content="# Leave Policy\n\nPTO carryover is allowed up to 3 days.",
            metadata={"kb_id": "default"},
        ),
        [
            Chunk(
                chunk_id="doc-2:0",
                doc_id="doc-2",
                chunk_index=0,
                text="PTO carryover is allowed up to 3 days.",
                source_path="data/raw/policy_b.md",
                title="Policy B",
                metadata={"kb_id": "default"},
            )
        ],
    )

    topic_page = (tmp_path / "wiki" / "topics" / "default--leave-policy.md").read_text(
        encoding="utf-8"
    )

    assert "## Key Facts" in topic_page
    assert "up to 5 days" in topic_page
    assert "up to 3 days" in topic_page
    # No hardcoded semantic conflict heuristics: two sources disagreeing on a
    # number is not machine-detectable as a "conflict" without semantic
    # judgment. Version conflicts are expressed structurally by the version
    # ledger (source_key groups), not guessed from number sets.
    assert "conflicts with" not in topic_page


def test_wiki_compiler_scope_id_is_collision_free_for_cjk() -> None:
    # Pure-CJK kb_ids previously all collapsed onto the shared "default"
    # scope, merging unrelated KBs into one index page — a cross-tenant leak.
    assert WikiCompiler.scope_id("default") == "default"
    assert WikiCompiler.scope_id("finance") == "finance"
    assert WikiCompiler.scope_id("财务库") != WikiCompiler.scope_id("人事库")
    assert WikiCompiler.scope_id("财务库") != "default"
    assert WikiCompiler.scope_id("财务库").startswith("kb-")


def test_wiki_compiler_separates_cjk_kbs_into_distinct_scopes(tmp_path) -> None:
    compiler = WikiCompiler(tmp_path / "wiki")
    for kb_id, doc_id, secret in (
        ("财务库", "fin-doc", "quarterly revenue figures"),
        ("人事库", "hr-doc", "employee salary bands"),
    ):
        compiler.upsert_document(
            Document(
                doc_id=doc_id,
                source_path=f"uploads/{kb_id}/{doc_id}.md",
                title=f"{doc_id} title",
                content=f"# {doc_id}\n\n{secret}",
                metadata={"kb_id": kb_id},
            ),
            [
                Chunk(
                    chunk_id=f"{doc_id}:0",
                    doc_id=doc_id,
                    chunk_index=0,
                    text=secret,
                    source_path=f"uploads/{kb_id}/{doc_id}.md",
                    title=doc_id,
                    metadata={"kb_id": kb_id},
                )
            ],
        )

    index_files = sorted(path.name for path in (tmp_path / "wiki" / "indexes").glob("*.md"))
    assert len(index_files) == 2
    assert "default.md" not in index_files
    # Neither scoped index page may leak the other KB's content.
    for name in index_files:
        content = (tmp_path / "wiki" / "indexes" / name).read_text(encoding="utf-8")
        assert "salary bands" not in content or "revenue figures" not in content


def test_wiki_compiler_remove_document_logs_deletion(tmp_path) -> None:
    compiler = WikiCompiler(tmp_path / "wiki")
    compiler.upsert_document(
        Document(
            doc_id="doc-1",
            source_path="data/raw/handbook.md",
            title="Employee Handbook",
            content="# Leave Policy\n\nRules.",
            metadata={"kb_id": "default"},
        ),
        [],
    )
    compiler.remove_document("doc-1")

    log_page = (tmp_path / "wiki" / "log.md").read_text(encoding="utf-8")
    assert "delete | doc-1" in log_page
    assert not (tmp_path / "wiki" / "sources" / "doc-1.md").exists()


def test_wiki_compiler_incremental_upsert_rewrites_only_changed_pages(tmp_path) -> None:
    # The version-chain bookkeeping must not bump mtimes of unaffected pages:
    # the query-side searcher diff-syncs on mtime, and a full rewrite turned
    # every ingest into a re-index storm over the whole corpus.
    compiler = WikiCompiler(tmp_path / "wiki")
    docs = [
        Document(
            doc_id=f"doc-{i}",
            source_path=f"data/raw/doc-{i}.md",
            title=f"Doc {i}",
            content=f"# Doc {i}\n\nBody {i}.",
            metadata={"kb_id": "default", "source_key": "unrelated", "version": "v1"},
        )
        for i in range(5)
    ]
    for doc in docs:
        compiler.upsert_document(doc, [])
    mtimes = {
        p.name: p.stat().st_mtime_ns
        for p in sorted((tmp_path / "wiki" / "sources").glob("*.md"))
    }

    # Upsert an unrelated document: existing pages must not be rewritten.
    compiler.upsert_document(
        Document(
            doc_id="doc-new",
            source_path="data/raw/doc-new.md",
            title="Doc New",
            content="# Doc New\n\nBody new.",
            metadata={"kb_id": "default", "source_key": "other", "version": "v1"},
        ),
        [],
    )
    mtimes_after = {
        p.name: p.stat().st_mtime_ns
        for p in sorted((tmp_path / "wiki" / "sources").glob("*.md"))
    }
    for name, mtime in mtimes.items():
        assert mtimes_after[name] == mtime, f"{name} was needlessly rewritten"


def test_wiki_compiler_corrupt_source_page_fails_loud(tmp_path) -> None:
    compiler = WikiCompiler(tmp_path / "wiki")
    compiler.upsert_document(
        Document(
            doc_id="doc-1",
            source_path="data/raw/handbook.md",
            title="Employee Handbook",
            content="# Leave Policy\n\nRules.",
            metadata={"kb_id": "default", "source_key": "policy"},
        ),
        [],
    )
    # Simulate an out-of-band corruption (no frontmatter left).
    (tmp_path / "wiki" / "sources" / "doc-1.md").write_text(
        "torn content without frontmatter", encoding="utf-8"
    )
    with pytest.raises(RetrievalError, match="unreadable frontmatter"):
        compiler.upsert_document(
            Document(
                doc_id="doc-2",
                source_path="data/raw/other.md",
                title="Other",
                content="# Other\n\nRules.",
                metadata={"kb_id": "default", "source_key": "other"},
            ),
            [],
        )
