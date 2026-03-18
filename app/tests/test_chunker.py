from app.ingestion.chunker import build_chunks, split_text


def test_split_text_respects_chunk_size() -> None:
    text = "A" * 500 + "\n\n" + "B" * 500 + "\n\n" + "C" * 500
    chunks = split_text(text, chunk_size=800, overlap=120)
    assert chunks
    assert all(len(chunk) <= 800 for chunk in chunks)


def test_build_chunks_indexes_are_stable() -> None:
    chunks = build_chunks("doc-1", "/tmp/source.txt", "source", "hello\n\nworld", 20, 5)
    assert [chunk.chunk_index for chunk in chunks] == [0]
