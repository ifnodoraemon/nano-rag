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
    assert "ingest | Employee Handbook" in log_page
    assert "compiled knowledge layer" in schema_page


def test_wiki_compiler_topic_page_aggregates_facts_and_conflicts(tmp_path) -> None:
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
            content="# Leave Policy\n\nPTO carryover is not allowed.",
            metadata={"kb_id": "default"},
        ),
        [
            Chunk(
                chunk_id="doc-2:0",
                doc_id="doc-2",
                chunk_index=0,
                text="PTO carryover is not allowed.",
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
    assert "not allowed" in topic_page
    assert "## Potential Conflicts" in topic_page
    assert "- conflicting" in topic_page
    assert "conflicts with" in topic_page
