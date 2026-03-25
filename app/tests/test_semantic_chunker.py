from app.ingestion.semantic_chunker import SemanticChunker, SemanticChunkerConfig


def test_semantic_chunker_config_defaults() -> None:
    config = SemanticChunkerConfig.from_env()
    assert config.min_chunk_size == 100
    assert config.max_chunk_size == 1000
    assert config.overlap_sentences == 1


def test_semantic_chunker_returns_chunks() -> None:
    chunker = SemanticChunker()
    text = "This is sentence one. This is sentence two. This is sentence three."
    chunks = chunker.chunk(text, "doc1", "test.txt", "Test")
    assert len(chunks) >= 1
    assert all(chunk.doc_id == "doc1" for chunk in chunks)
    assert all(chunk.source_path == "test.txt" for chunk in chunks)


def test_semantic_chunker_empty_text() -> None:
    chunker = SemanticChunker()
    chunks = chunker.chunk("", "doc1", "test.txt", "Test")
    assert len(chunks) == 0


def test_semantic_chunker_respects_max_chunk_size() -> None:
    config = SemanticChunkerConfig(max_chunk_size=50)
    chunker = SemanticChunker(config=config)
    text = "This is a longer sentence that should be split. And another one here."
    chunks = chunker.chunk(text, "doc1", "test.txt", "Test")
    for chunk in chunks:
        assert len(chunk.text) <= config.max_chunk_size or len(chunks) == 1
