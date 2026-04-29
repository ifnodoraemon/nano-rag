from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from app.core.exceptions import ModelGatewayError
from app.model_client.multimodal_embedding import (
    FileItem,
    GeminiMultimodalEmbedding,
    ImageItem,
    TextItem,
)


def _make_config(
    *,
    api_key: str = "test-key",
    dimension: int = 1536,
    base_url: str = "https://generativelanguage.googleapis.com",
    mode: str = "live",
) -> SimpleNamespace:
    return SimpleNamespace(
        models={
            "embedding": {
                "default_alias": "gemini-embedding-2-preview",
                "dimension": dimension,
                "base_url": base_url,
                "api_key": api_key,
            }
        },
        settings={"timeout": {"embeddings_seconds": 30}},
        gateway_mode=mode,
    )


def _install_transport(
    client: GeminiMultimodalEmbedding, handler
) -> list[httpx.Request]:
    captured: list[httpx.Request] = []

    def _wrap(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return handler(request)

    client._client = httpx.AsyncClient(transport=httpx.MockTransport(_wrap))
    return captured


def _ok_response(values: list[float]) -> httpx.Response:
    return httpx.Response(200, json={"embeddings": [{"values": values}]})


@pytest.mark.asyncio
async def test_text_only_payload_shape() -> None:
    config = _make_config(dimension=8)
    client = GeminiMultimodalEmbedding(config)
    captured = _install_transport(
        client, lambda req: _ok_response([0.1] * 8)
    )

    vector = await client.embed_one([TextItem("hello world")])

    assert vector == [0.1] * 8
    assert len(captured) == 1
    assert captured[0].url.path.endswith(
        "/v1beta/models/gemini-embedding-2-preview:embedContent"
    )
    assert captured[0].headers["x-goog-api-key"] == "test-key"
    body = json.loads(captured[0].content)
    assert body["output_dimensionality"] == 8
    assert body["content"]["parts"] == [{"text": "hello world"}]


@pytest.mark.asyncio
async def test_image_text_interleaved_payload() -> None:
    config = _make_config(dimension=4)
    client = GeminiMultimodalEmbedding(config)
    captured = _install_transport(
        client, lambda req: _ok_response([0.0, 0.0, 0.0, 0.0])
    )

    image_bytes = b"\x89PNG\r\nfakebytes"
    await client.embed_one([TextItem("describe"), ImageItem(image_bytes, "image/png")])

    body = json.loads(captured[0].content)
    parts = body["content"]["parts"]
    assert parts[0] == {"text": "describe"}
    assert parts[1]["inline_data"]["mime_type"] == "image/png"
    assert parts[1]["inline_data"]["data"] == base64.b64encode(image_bytes).decode()


@pytest.mark.asyncio
async def test_file_item_reads_bytes(tmp_path: Path) -> None:
    config = _make_config(dimension=2)
    client = GeminiMultimodalEmbedding(config)
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return _ok_response([0.5, 0.5])

    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    sample = tmp_path / "sample.png"
    sample.write_bytes(b"PNGBYTES")
    await client.embed_one([FileItem(sample, "image/png")])

    body = json.loads(captured[0].content)
    inline = body["content"]["parts"][0]["inline_data"]
    assert inline["mime_type"] == "image/png"
    assert inline["data"] == base64.b64encode(b"PNGBYTES").decode()


@pytest.mark.asyncio
async def test_missing_api_key_raises() -> None:
    config = _make_config(api_key="")
    client = GeminiMultimodalEmbedding(config)
    # Live mode requires a key; mock fallback bypasses, so flip explicitly
    client.api_key = ""
    with pytest.raises(ModelGatewayError, match="EMBEDDING_API_KEY"):
        await client.embed_one([TextItem("hi")])


@pytest.mark.asyncio
async def test_http_error_wrapped() -> None:
    config = _make_config()
    client = GeminiMultimodalEmbedding(config)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limit")

    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    with pytest.raises(ModelGatewayError, match="429"):
        await client.embed_one([TextItem("hi")])


@pytest.mark.asyncio
async def test_empty_response_raises() -> None:
    config = _make_config()
    client = GeminiMultimodalEmbedding(config)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [{}]})

    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    with pytest.raises(ModelGatewayError, match="no values"):
        await client.embed_one([TextItem("hi")])


@pytest.mark.asyncio
async def test_mock_mode_returns_deterministic_vector() -> None:
    config = _make_config(mode="mock", api_key="", dimension=16)
    client = GeminiMultimodalEmbedding(config)

    v1 = await client.embed_one([TextItem("alpha beta")])
    v2 = await client.embed_one([TextItem("alpha beta")])
    v3 = await client.embed_one([TextItem("gamma delta")])

    assert len(v1) == 16
    assert v1 == v2
    assert v1 != v3


@pytest.mark.asyncio
async def test_embed_batch_preserves_order() -> None:
    config = _make_config(mode="mock", api_key="", dimension=8)
    client = GeminiMultimodalEmbedding(config)

    batches = [[TextItem(f"item-{i}")] for i in range(4)]
    vectors = await client.embed_batch(batches)

    assert len(vectors) == 4
    assert all(len(v) == 8 for v in vectors)
    # Same input twice must produce same vector (determinism)
    repeat = await client.embed_batch([batches[0]])
    assert repeat[0] == vectors[0]
