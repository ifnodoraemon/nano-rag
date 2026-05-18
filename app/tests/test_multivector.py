from app.retrieval.multivector import (
    ColPaliHttpMultiVectorProvider,
    LightweightMultiVectorProvider,
    MultiVectorStore,
    attach_chunk_multivectors,
    late_interaction_score,
    multivector_provider_from_config,
)
from app.schemas.chunk import Chunk


def test_attach_chunk_multivectors_adds_lightweight_vectors() -> None:
    provider = LightweightMultiVectorProvider()
    chunk = Chunk(
        chunk_id="page-image",
        doc_id="doc",
        chunk_index=0,
        text="",
        source_path="/tmp/contract.pdf",
        title="Contract page image",
        metadata={
            "kb_id": "default",
            "chunk_kind": "rendered_page_image",
            "chunk_strategy": "rendered_page_image",
            "attachment_scope": "page_image",
        },
        modality="image",
        media_uri="/tmp/page.png",
        mime_type="image/png",
    )

    enriched = attach_chunk_multivectors(chunk, provider=provider)

    assert enriched.metadata["multi_vector_model"] == "lightweight-hash-v1"
    assert enriched.metadata["multi_vector_dim"] == 32
    assert len(enriched.metadata["multi_vector"]) > 1


def test_late_interaction_scores_visual_hint_match_higher() -> None:
    provider = LightweightMultiVectorProvider()
    visual = attach_chunk_multivectors(
        Chunk(
            chunk_id="visual",
            doc_id="doc",
            chunk_index=0,
            text="",
            source_path="/tmp/contract.pdf",
            title="Contract page image",
            metadata={"kb_id": "default", "chunk_kind": "rendered_page_image"},
            modality="image",
            media_uri="/tmp/page.png",
            mime_type="image/png",
        ),
        provider=provider,
    )
    text = attach_chunk_multivectors(
        Chunk(
            chunk_id="text",
            doc_id="doc",
            chunk_index=1,
            text="plain contract paragraph",
            source_path="/tmp/contract.pdf",
            title="Contract",
            metadata={"kb_id": "default"},
        ),
        provider=provider,
    )

    assert late_interaction_score("contract page image", visual, provider=provider) > late_interaction_score(
        "contract page image",
        text,
        provider=provider,
    )


def test_multivector_store_keeps_vectors_out_of_chunk_metadata(tmp_path) -> None:
    store = MultiVectorStore(tmp_path / "multivectors")
    chunk = Chunk(
        chunk_id="page-image",
        doc_id="doc",
        chunk_index=0,
        text="",
        source_path="/tmp/contract.pdf",
        title="Contract page image",
        metadata={"kb_id": "default", "chunk_kind": "rendered_page_image"},
        modality="image",
        media_uri="/tmp/page.png",
        mime_type="image/png",
    )

    provider = LightweightMultiVectorProvider()
    enriched = attach_chunk_multivectors(chunk, provider=provider, store=store)

    assert "multi_vector" not in enriched.metadata
    assert enriched.metadata["multi_vector_ref"].startswith("mv-")
    assert enriched.metadata["multi_vector_count"] > 0
    assert store.get(enriched.metadata["multi_vector_ref"])
    assert late_interaction_score("contract page image", enriched, provider=provider, store=store) > 0


def test_multivector_store_refs_are_content_addressed(tmp_path) -> None:
    store = MultiVectorStore(tmp_path / "multivectors")
    first = attach_chunk_multivectors(
        Chunk(
            chunk_id="stable-chunk",
            doc_id="doc",
            chunk_index=0,
            text="first version",
            source_path="/tmp/policy.txt",
            title="Policy",
            metadata={"kb_id": "default"},
        ),
        provider=LightweightMultiVectorProvider(),
        store=store,
    )
    second = attach_chunk_multivectors(
        Chunk(
            chunk_id="stable-chunk",
            doc_id="doc",
            chunk_index=0,
            text="second version",
            source_path="/tmp/policy.txt",
            title="Policy",
            metadata={"kb_id": "default"},
        ),
        provider=LightweightMultiVectorProvider(),
        store=store,
    )

    assert first.metadata["multi_vector_ref"] != second.metadata["multi_vector_ref"]
    assert store.get(first.metadata["multi_vector_ref"])
    assert store.get(second.metadata["multi_vector_ref"])

    store.delete_refs({second.metadata["multi_vector_ref"]})

    assert store.get(first.metadata["multi_vector_ref"])
    assert not store.get(second.metadata["multi_vector_ref"])


def test_multivector_provider_defaults_to_disabled() -> None:
    config = type("Config", (), {"models": {}})()

    assert multivector_provider_from_config(config) is None


def test_colpali_http_provider_extracts_real_patch_vectors(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"\x89PNGfake")

    requests = []

    class FakeResponse:
        status_code = 200
        text = ""

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"vectors": [[0.1, 0.2], [0.3, 0.4]]}

    class FakeClient:
        def __init__(self, timeout):  # noqa: ANN001
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

        def post(self, url, headers, json):  # noqa: ANN001
            requests.append({"url": url, "headers": headers, "json": json})
            return FakeResponse()

    monkeypatch.setattr("app.retrieval.multivector.httpx.Client", FakeClient)
    provider = ColPaliHttpMultiVectorProvider(
        model_name="vidore/colqwen2-v1.0-hf",
        base_url="http://colpali:8080",
        api_key="secret",
    )
    chunk = Chunk(
        chunk_id="page",
        doc_id="doc",
        chunk_index=0,
        text="",
        source_path="contract.pdf",
        title="Contract page",
        metadata={"chunk_kind": "rendered_page_image", "page_number": 3},
        modality="image",
        media_uri=str(image_path),
        mime_type="image/png",
    )

    vectors = provider.embed_chunk(chunk)
    query_vectors = provider.embed_query("where is the signature")

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert query_vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert requests[0]["url"] == "http://colpali:8080/embed"
    assert requests[0]["headers"]["Authorization"] == "Bearer secret"
    assert requests[0]["json"]["input_type"] == "document"
    assert requests[0]["json"]["image"].startswith("data:image/png;base64,")
    assert requests[1]["json"]["input_type"] == "query"
