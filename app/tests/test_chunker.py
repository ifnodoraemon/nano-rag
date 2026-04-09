from app.ingestion.chunker import build_chunks, split_text


def test_split_text_respects_chunk_size() -> None:
    text = "A" * 500 + "\n\n" + "B" * 500 + "\n\n" + "C" * 500
    chunks = split_text(text, chunk_size=800, overlap=120)
    assert chunks
    assert all(len(chunk) <= 800 for chunk in chunks)


def test_build_chunks_indexes_are_stable() -> None:
    chunks = build_chunks("doc-1", "/tmp/source.txt", "source", "hello\n\nworld", 20, 5)
    assert [chunk.chunk_index for chunk in chunks] == [0]


def test_build_chunks_adds_section_and_parent_metadata() -> None:
    text = (
        "# Leave Policy\n\n"
        "Effective Date: 2026-01-15\n"
        "Version: v2.1\n"
        "Department: HR\n\n"
        "Carryover is allowed.\n\n"
        "## Expense Rules\n\n"
        "Claims must be filed within 15 days."
    )

    chunks = build_chunks(
        "doc-1",
        "/tmp/policy.md",
        "Policy",
        text,
        120,
        20,
        metadata={
            "doc_type": "policy",
            "source_key": "policy",
            "effective_date": "2026-01-15",
            "version": "v2.1",
            "department": "HR",
        },
    )

    assert len(chunks) >= 2
    assert chunks[0].metadata["parent_chunk_id"] == "doc-1:parent:0"
    assert chunks[0].metadata["section_path"] == ["Policy", "Leave Policy"]
    assert chunks[0].metadata["doc_type"] == "policy"
    assert chunks[0].metadata["source_key"] == "policy"
    assert chunks[0].metadata["effective_date"] == "2026-01-15"
    assert chunks[0].metadata["version"] == "v2.1"
    assert chunks[-1].metadata["section_path"] == [
        "Policy",
        "Leave Policy",
        "Expense Rules",
    ]
