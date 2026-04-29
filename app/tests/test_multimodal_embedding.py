from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from app.core.exceptions import ModelGatewayError
from app.model_client.multimodal_embedding import (
    AudioItem,
    DashScopeMultimodalEmbedding,
    FileItem,
    GeminiMultimodalEmbedding,
    ImageItem,
    TextItem,
    VLLMMultimodalEmbedding,
    VideoItem,
    create_multimodal_embedding,
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
async def test_embed_batch_preserves_order() -> None:
    config = _make_config(dimension=8)
    client = GeminiMultimodalEmbedding(config)
    values_by_call = [[float(i)] * 8 for i in range(4)]

    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return _ok_response(values_by_call.pop(0))

    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    batches = [[TextItem(f"item-{i}")] for i in range(4)]
    vectors = await client.embed_batch(batches)

    assert len(vectors) == 4
    assert all(len(v) == 8 for v in vectors)
    assert vectors[0] == [0.0] * 8
    assert vectors[3] == [3.0] * 8


def _dashscope_config(
    *,
    api_key: str = "ds-key",
    dimension: int = 1536,
    base_url: str = "https://dashscope.aliyuncs.com",
    mode: str = "live",
) -> SimpleNamespace:
    return SimpleNamespace(
        models={
            "embedding": {
                "default_alias": "multimodal-embedding-v1",
                "dimension": dimension,
                "base_url": base_url,
                "api_key": api_key,
                "provider": "dashscope",
            }
        },
        settings={"timeout": {"embeddings_seconds": 30}},
        gateway_mode=mode,
    )


def _vllm_config(
    *,
    base_url: str = "http://localhost:8001/v1",
    dimension: int = 1536,
    mode: str = "live",
) -> SimpleNamespace:
    return SimpleNamespace(
        models={
            "embedding": {
                "default_alias": "Qwen/Qwen3-VL-Embedding-8B",
                "dimension": dimension,
                "base_url": base_url,
                "provider": "vllm",
            }
        },
        settings={"timeout": {"embeddings_seconds": 30}},
        gateway_mode=mode,
    )


@pytest.mark.asyncio
async def test_dashscope_text_payload_shape() -> None:
    config = _dashscope_config(dimension=8)
    client = DashScopeMultimodalEmbedding(config)
    captured: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return httpx.Response(
            200,
            json={"output": {"embeddings": [{"embedding": [0.1] * 8}]}},
        )

    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    vector = await client.embed_one([TextItem("你好")])

    assert vector == [0.1] * 8
    assert captured[0].url.path.endswith(
        "/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"
    )
    assert captured[0].headers["Authorization"] == "Bearer ds-key"
    body = json.loads(captured[0].content)
    assert body["model"] == "multimodal-embedding-v1"
    assert body["input"]["contents"] == [{"text": "你好"}]
    assert body["parameters"]["dimension"] == 8


@pytest.mark.asyncio
async def test_dashscope_image_text_payload() -> None:
    config = _dashscope_config(dimension=4)
    client = DashScopeMultimodalEmbedding(config)
    captured: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return httpx.Response(
            200,
            json={"output": {"embeddings": [{"embedding": [0.0, 0.0, 0.0, 0.0]}]}},
        )

    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    image = b"\x89PNG\r\n"
    await client.embed_one([TextItem("logo"), ImageItem(image, "image/png")])

    body = json.loads(captured[0].content)
    contents = body["input"]["contents"]
    assert contents[0] == {"text": "logo"}
    assert contents[1]["image"].startswith("data:image/png;base64,")
    assert contents[1]["image"].endswith(
        base64.b64encode(image).decode()
    )


@pytest.mark.asyncio
async def test_dashscope_missing_key_raises() -> None:
    config = _dashscope_config(api_key="")
    client = DashScopeMultimodalEmbedding(config)
    client.api_key = ""
    with pytest.raises(ModelGatewayError, match="EMBEDDING_API_KEY"):
        await client.embed_one([TextItem("hi")])


@pytest.mark.asyncio
async def test_vllm_messages_style_payload() -> None:
    config = _vllm_config(dimension=4)
    client = VLLMMultimodalEmbedding(config)
    captured: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(req)
        return httpx.Response(
            200,
            json={"data": [{"embedding": [0.2, 0.3, 0.4, 0.5]}]},
        )

    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    image = b"PNGBYTES"
    vector = await client.embed_one(
        [TextItem("describe"), ImageItem(image, "image/png")]
    )

    assert vector == [0.2, 0.3, 0.4, 0.5]
    assert captured[0].url.path.endswith("/v1/embeddings")
    body = json.loads(captured[0].content)
    assert body["model"] == "Qwen/Qwen3-VL-Embedding-8B"
    assert body["input"][0]["role"] == "user"
    parts = body["input"][0]["content"]
    assert parts[0] == {"type": "text", "text": "describe"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_audio_video_items_handled(tmp_path: Path) -> None:
    """Audio/video items should serialize correctly across all providers."""
    audio = AudioItem(b"AUDIO", "audio/mp3")
    video = VideoItem(b"VIDEO", "video/mp4")

    gem_config = _make_config(dimension=4)
    gem = GeminiMultimodalEmbedding(gem_config)
    gem_part_audio = gem._build_part(audio)
    gem_part_video = gem._build_part(video)
    assert gem_part_audio["inline_data"]["mime_type"] == "audio/mp3"
    assert gem_part_video["inline_data"]["mime_type"] == "video/mp4"

    ds_config = _dashscope_config(dimension=4)
    ds = DashScopeMultimodalEmbedding(ds_config)
    assert "audio" in ds._build_content(audio)
    assert "video" in ds._build_content(video)

    vl_config = _vllm_config(dimension=4)
    vl = VLLMMultimodalEmbedding(vl_config)
    assert vl._build_part(audio)["type"] == "audio_url"
    assert vl._build_part(video)["type"] == "video_url"


def test_factory_selects_provider() -> None:
    gem_cfg = _make_config()
    gem_cfg.models["embedding"]["provider"] = "gemini"
    assert isinstance(create_multimodal_embedding(gem_cfg), GeminiMultimodalEmbedding)

    ds_cfg = _dashscope_config()
    assert isinstance(
        create_multimodal_embedding(ds_cfg), DashScopeMultimodalEmbedding
    )

    vl_cfg = _vllm_config()
    assert isinstance(create_multimodal_embedding(vl_cfg), VLLMMultimodalEmbedding)

    bad = _make_config()
    bad.models["embedding"]["provider"] = "nonsense"
    with pytest.raises(ModelGatewayError, match="unknown EMBEDDING_PROVIDER"):
        create_multimodal_embedding(bad)


def test_factory_env_overrides_config(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_config()
    cfg.models["embedding"]["provider"] = "gemini"
    monkeypatch.setenv("EMBEDDING_PROVIDER", "dashscope")
    assert isinstance(
        create_multimodal_embedding(cfg), DashScopeMultimodalEmbedding
    )
