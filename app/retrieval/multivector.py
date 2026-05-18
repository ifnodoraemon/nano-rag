from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from typing import Iterable

import httpx

from app.core.exceptions import ModelGatewayError
from app.schemas.chunk import Chunk


DEFAULT_MULTIVECTOR_DIM = 32
DEFAULT_MAX_CHUNK_VECTORS = 32
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]{1,4}")
VISUAL_HINTS = {
    "chart",
    "diagram",
    "embedded",
    "figure",
    "image",
    "layout",
    "logo",
    "page",
    "photo",
    "rendered",
    "scan",
    "signature",
    "stamp",
    "visual",
    "图",
    "图片",
    "图表",
    "扫描",
    "签名",
    "签章",
    "盖章",
    "印章",
    "页面",
    "版面",
}


class MultiVectorProvider(Protocol):
    model_name: str
    dim: int

    def embed_query(self, query: str) -> list[list[float]]: ...

    def embed_chunk(self, chunk: Chunk) -> list[list[float]]: ...

    def should_embed_chunk(self, chunk: Chunk) -> bool: ...


@dataclass(frozen=True)
class LightweightMultiVectorProvider:
    dim: int = DEFAULT_MULTIVECTOR_DIM
    max_chunk_vectors: int = DEFAULT_MAX_CHUNK_VECTORS
    model_name: str = "lightweight-hash-v1"

    def should_embed_chunk(self, chunk: Chunk) -> bool:  # noqa: ARG002
        return True

    def embed_query(self, query: str) -> list[list[float]]:
        return build_query_multivectors(query, dim=self.dim)

    def embed_chunk(self, chunk: Chunk) -> list[list[float]]:
        return build_chunk_multivectors(
            chunk,
            dim=self.dim,
            max_vectors=self.max_chunk_vectors,
        )


class ColPaliHttpMultiVectorProvider:
    def __init__(
        self,
        *,
        model_name: str,
        base_url: str,
        api_key: str = "",
        path: str = "/embed",
        dim: int = 128,
        timeout_seconds: int = 120,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.path = path if path.startswith("/") else f"/{path}"
        self.dim = dim
        self.timeout_seconds = timeout_seconds

    def should_embed_chunk(self, chunk: Chunk) -> bool:
        return _is_visual_multivector_chunk(chunk)

    def embed_query(self, query: str) -> list[list[float]]:
        payload = {
            "model": self.model_name,
            "input_type": "query",
            "query": query,
        }
        return self._post_vectors(payload)

    def embed_chunk(self, chunk: Chunk) -> list[list[float]]:
        payload = self._chunk_payload(chunk)
        if payload is None:
            return []
        return self._post_vectors(payload)

    def _chunk_payload(self, chunk: Chunk) -> dict[str, object] | None:
        media_path = _chunk_media_path(chunk)
        metadata = chunk.metadata or {}
        payload: dict[str, object] = {
            "model": self.model_name,
            "input_type": "document",
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "title": chunk.title,
            "text": chunk.text,
            "metadata": {
                "source_path": chunk.source_path,
                "page_number": metadata.get("page_number"),
                "chunk_kind": metadata.get("chunk_kind"),
                "chunk_strategy": metadata.get("chunk_strategy"),
                "attachment_scope": metadata.get("attachment_scope"),
            },
        }
        if media_path is None:
            return payload
        mime_type = chunk.mime_type or _guess_media_mime(media_path)
        payload["mime_type"] = mime_type
        if mime_type.startswith("image/"):
            payload["image"] = _data_url(media_path, mime_type)
        else:
            payload["file"] = _data_url(media_path, mime_type)
        return payload

    def _post_vectors(self, payload: dict[str, object]) -> list[list[float]]:
        if not self.base_url:
            raise ModelGatewayError("MULTIVECTOR_API_BASE_URL must be configured.")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(
                    f"{self.base_url}{self.path}",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ModelGatewayError("multivector provider timeout") from exc
        except httpx.HTTPStatusError as exc:
            raise ModelGatewayError(
                f"multivector provider request failed: {exc.response.status_code} "
                f"{exc.response.text.strip()}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ModelGatewayError(f"multivector provider connection failed: {exc}") from exc
        vectors = _extract_provider_vectors(response.json())
        self._update_dim(vectors)
        return vectors

    def _update_dim(self, vectors: list[list[float]]) -> None:
        if vectors and vectors[0]:
            self.dim = len(vectors[0])


class TransformersColVisionMultiVectorProvider:
    def __init__(
        self,
        *,
        model_name: str,
        family: str = "colqwen2",
        dim: int = 128,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_vectors: int = 0,
    ) -> None:
        self.model_name = model_name
        self.family = family.lower()
        self.dim = dim
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_vectors = max_vectors
        self._lock = threading.Lock()
        self._model = None
        self._processor = None
        self._torch = None
        self._image_cls = None

    def should_embed_chunk(self, chunk: Chunk) -> bool:
        media_path = _chunk_media_path(chunk)
        return _is_visual_multivector_chunk(chunk) and media_path is not None and _is_image_path(media_path, chunk.mime_type)

    def embed_query(self, query: str) -> list[list[float]]:
        model, processor, torch, _ = self._load()
        inputs = processor(text=[query], return_tensors="pt").to(_model_device(model))
        with torch.no_grad():
            output = model(**inputs)
        vectors = _tensor_to_vectors(getattr(output, "embeddings", output))
        self._update_dim(vectors)
        return vectors

    def embed_chunk(self, chunk: Chunk) -> list[list[float]]:
        media_path = _chunk_media_path(chunk)
        if media_path is None or not _is_image_path(media_path, chunk.mime_type):
            return []
        model, processor, torch, image_cls = self._load()
        image = image_cls.open(media_path).convert("RGB")
        inputs = processor(images=[image], return_tensors="pt").to(_model_device(model))
        with torch.no_grad():
            output = model(**inputs)
        vectors = _tensor_to_vectors(getattr(output, "embeddings", output))
        if self.max_vectors > 0:
            vectors = vectors[: self.max_vectors]
        self._update_dim(vectors)
        return vectors

    def _load(self):
        with self._lock:
            if self._model is not None and self._processor is not None:
                return self._model, self._processor, self._torch, self._image_cls
            try:
                import torch
                from PIL import Image
                from transformers.utils.import_utils import is_flash_attn_2_available
            except ImportError as exc:
                raise ModelGatewayError(
                    "Real ColPali/ColQwen multivector provider requires optional "
                    "dependencies: torch, transformers, accelerate and pillow. "
                    "Install requirements-multivector.txt in the GPU image."
                ) from exc
            try:
                if self.family in {"colpali", "pali"}:
                    from transformers import ColPaliForRetrieval, ColPaliProcessor

                    model_cls = ColPaliForRetrieval
                    processor_cls = ColPaliProcessor
                else:
                    from transformers import ColQwen2ForRetrieval, ColQwen2Processor

                    model_cls = ColQwen2ForRetrieval
                    processor_cls = ColQwen2Processor
            except ImportError as exc:
                raise ModelGatewayError(
                    "Installed transformers version does not expose ColPali/ColQwen2 "
                    "retrieval classes. Use transformers>=4.53 for ColQwen2."
                ) from exc
            kwargs: dict[str, object] = {"device_map": self.device_map}
            dtype = _torch_dtype(torch, self.torch_dtype)
            if dtype is not None:
                kwargs["torch_dtype"] = dtype
            if self.family in {"colqwen2", "colqwen", "qwen"}:
                kwargs["attn_implementation"] = (
                    "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
                )
            self._model = model_cls.from_pretrained(self.model_name, **kwargs).eval()
            self._processor = processor_cls.from_pretrained(self.model_name)
            self._torch = torch
            self._image_cls = Image
            return self._model, self._processor, self._torch, self._image_cls

    def _update_dim(self, vectors: list[list[float]]) -> None:
        if vectors and vectors[0]:
            self.dim = len(vectors[0])


class MultiVectorStore:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir

    def upsert(
        self,
        *,
        chunk: Chunk,
        vectors: list[list[float]],
        model_name: str,
        dim: int,
    ) -> str:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        ref = self._ref_for_vectors(
            chunk_id=chunk.chunk_id,
            vectors=vectors,
            model_name=model_name,
            dim=dim,
        )
        target = self.root_dir / f"{ref}.json"
        tmp = target.with_suffix(".json.tmp")
        payload = {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "model": model_name,
            "dim": dim,
            "vectors": vectors,
        }
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(str(tmp), str(target))
        return ref

    def delete_refs(self, refs: set[str]) -> None:
        for ref in refs:
            if not ref.startswith("mv-"):
                continue
            (self.root_dir / f"{ref}.json").unlink(missing_ok=True)

    def get(self, ref: str) -> list[list[float]]:
        target = self.root_dir / f"{ref}.json"
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        vectors = payload.get("vectors")
        if not isinstance(vectors, list):
            return []
        return _coerce_vectors(vectors)

    @staticmethod
    def _ref_for_vectors(
        *,
        chunk_id: str,
        vectors: list[list[float]],
        model_name: str,
        dim: int,
    ) -> str:
        identity = {
            "chunk_id": chunk_id,
            "dim": dim,
            "model": model_name,
            "vectors": vectors,
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return f"mv-{digest[:32]}"


def multivector_store_from_config(config: object) -> MultiVectorStore:
    try:
        parsed_dir = getattr(config, "parsed_dir")
    except Exception:
        parsed_dir = Path(os.getenv("PARSED_OUTPUT_DIR", "/tmp/nano-rag-parsed"))
    if not isinstance(parsed_dir, Path):
        parsed_dir = Path(str(parsed_dir))
    return MultiVectorStore(parsed_dir / "multivectors")


def multivector_provider_from_config(config: object) -> MultiVectorProvider | None:
    section = getattr(config, "models", {}).get("multivector", {})
    provider = str(
        os.getenv("MULTIVECTOR_PROVIDER")
        or section.get("provider")
        or ""
    ).strip().lower()
    if provider in {"", "disabled", "none", "false", "off"}:
        return None
    model_name = str(
        os.getenv("MULTIVECTOR_MODEL_ALIAS")
        or section.get("default_alias")
        or "vidore/colqwen2-v1.0-hf"
    )
    dim = _int_setting("MULTIVECTOR_DIM", section.get("dimension"), 128)
    if provider in {"colpali-http", "colqwen-http", "http"}:
        return ColPaliHttpMultiVectorProvider(
            model_name=model_name,
            base_url=str(
                os.getenv("MULTIVECTOR_API_BASE_URL")
                or section.get("base_url")
                or ""
            ),
            api_key=str(
                os.getenv("MULTIVECTOR_API_KEY")
                or section.get("api_key")
                or ""
            ),
            path=str(
                os.getenv("MULTIVECTOR_API_PATH")
                or section.get("path")
                or "/embed"
            ),
            dim=dim,
            timeout_seconds=_int_setting(
                "MULTIVECTOR_TIMEOUT_SECONDS",
                section.get("timeout_seconds"),
                120,
            ),
        )
    if provider in {"colpali", "colqwen", "colqwen2", "transformers"}:
        family = str(
            os.getenv("MULTIVECTOR_MODEL_FAMILY")
            or section.get("family")
            or ("colpali" if "colpali" in model_name.lower() else "colqwen2")
        )
        return TransformersColVisionMultiVectorProvider(
            model_name=model_name,
            family=family,
            dim=dim,
            device_map=str(
                os.getenv("MULTIVECTOR_DEVICE_MAP")
                or section.get("device_map")
                or "auto"
            ),
            torch_dtype=str(
                os.getenv("MULTIVECTOR_TORCH_DTYPE")
                or section.get("torch_dtype")
                or "auto"
            ),
            max_vectors=_int_setting(
                "MULTIVECTOR_MAX_PATCH_VECTORS",
                section.get("max_patch_vectors"),
                0,
            ),
        )
    if provider == "lightweight" and _lightweight_multivectors_allowed():
        return LightweightMultiVectorProvider(dim=dim)
    if provider == "lightweight":
        raise ModelGatewayError(
            "MULTIVECTOR_PROVIDER=lightweight is only allowed when "
            "RAG_ALLOW_LIGHTWEIGHT_MULTIVECTOR=true. Configure colqwen2/colpali "
            "or colpali-http for production."
        )
    raise ModelGatewayError(
        f"unknown MULTIVECTOR_PROVIDER '{provider}'. Supported: colqwen2, colpali, colpali-http."
    )


def build_query_multivectors(query: str, *, dim: int = DEFAULT_MULTIVECTOR_DIM) -> list[list[float]]:
    return [_token_vector(token, dim=dim) for token in _tokens(query)[:DEFAULT_MAX_CHUNK_VECTORS]]


def build_chunk_multivectors(
    chunk: Chunk,
    *,
    dim: int = DEFAULT_MULTIVECTOR_DIM,
    max_vectors: int = DEFAULT_MAX_CHUNK_VECTORS,
) -> list[list[float]]:
    metadata = chunk.metadata or {}
    tokens = list(_tokens(chunk.text))
    tokens.extend(_metadata_tokens(metadata))
    tokens.extend(_tokens(chunk.title or ""))
    tokens.extend(_tokens(chunk.source_path))
    if chunk.modality in {"image", "document"}:
        tokens.extend(VISUAL_HINTS)
    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
        if len(deduped) >= max_vectors:
            break
    return [_token_vector(token, dim=dim) for token in deduped]


def attach_chunk_multivectors(
    chunk: Chunk,
    *,
    provider: MultiVectorProvider | None = None,
    store: MultiVectorStore | None = None,
    inline: bool | None = None,
) -> Chunk:
    if provider is None:
        return chunk
    metadata = chunk.metadata or {}
    if metadata.get("multi_vector") or metadata.get("multi_vector_ref"):
        return chunk
    if not _provider_should_embed(provider, chunk):
        return chunk
    vectors = provider.embed_chunk(chunk)
    if not vectors:
        return chunk
    if inline is None:
        inline = _inline_multivectors_enabled()
    update = {
        **metadata,
        "multi_vector_model": provider.model_name,
        "multi_vector_dim": provider.dim,
        "multi_vector_count": len(vectors),
    }
    if store is not None:
        update["multi_vector_ref"] = store.upsert(
            chunk=chunk,
            vectors=vectors,
            model_name=provider.model_name,
            dim=provider.dim,
        )
    if inline or store is None:
        update["multi_vector"] = vectors
    return chunk.model_copy(
        update={
            "metadata": update
        }
    )


def late_interaction_score(
    query: str,
    chunk: Chunk,
    *,
    provider: MultiVectorProvider | None = None,
    store: MultiVectorStore | None = None,
) -> float:
    if provider is None:
        return 0.0
    query_vectors = provider.embed_query(query)
    if not query_vectors:
        return 0.0
    chunk_vectors = _metadata_multivectors(chunk)
    if not chunk_vectors and store is not None:
        ref = (chunk.metadata or {}).get("multi_vector_ref")
        if isinstance(ref, str) and ref:
            chunk_vectors = store.get(ref)
    if not chunk_vectors:
        if not _provider_should_embed(provider, chunk):
            return 0.0
        chunk_vectors = provider.embed_chunk(chunk)
    if not chunk_vectors:
        return 0.0
    scores = []
    for query_vector in query_vectors:
        scores.append(max(_cosine(query_vector, chunk_vector) for chunk_vector in chunk_vectors))
    return round(sum(scores) / len(scores), 6)


def _provider_should_embed(provider: MultiVectorProvider, chunk: Chunk) -> bool:
    checker = getattr(provider, "should_embed_chunk", None)
    if callable(checker):
        return bool(checker(chunk))
    return _is_visual_multivector_chunk(chunk)


def _metadata_multivectors(chunk: Chunk) -> list[list[float]]:
    raw = (chunk.metadata or {}).get("multi_vector")
    if not isinstance(raw, list):
        return []
    return _coerce_vectors(raw)


def _coerce_vectors(raw: list[object]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for item in raw:
        if not isinstance(item, list):
            continue
        try:
            vector = [float(value) for value in item]
        except (TypeError, ValueError):
            continue
        if vector:
            vectors.append(vector)
    return vectors


def _metadata_tokens(metadata: dict[str, object]) -> Iterable[str]:
    keys = (
        "chunk_kind",
        "chunk_strategy",
        "source_modality",
        "attachment_scope",
        "doc_type",
        "source_file_name",
        "source_suffix",
        "mime_type",
        "definition_term",
        "clause_title",
    )
    for key in keys:
        value = metadata.get(key)
        if value is None:
            continue
        yield from _tokens(str(value))
    page_number = metadata.get("page_number")
    if page_number is not None:
        yield f"page_{page_number}"


def _tokens(value: str) -> list[str]:
    tokens = [match.group(0).casefold() for match in TOKEN_RE.finditer(value)]
    expanded: list[str] = []
    for token in tokens:
        expanded.append(token)
        if len(token) > 4 and any("\u4e00" <= char <= "\u9fff" for char in token):
            expanded.extend(token[index : index + 2] for index in range(0, len(token) - 1))
    return [token for token in expanded if token.strip()]


def _token_vector(token: str, *, dim: int) -> list[float]:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    values = []
    for index in range(dim):
        byte = digest[index % len(digest)]
        values.append((byte / 127.5) - 1.0)
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return values
    return [round(value / norm, 6) for value in values]


def _cosine(lhs: list[float], rhs: list[float]) -> float:
    size = min(len(lhs), len(rhs))
    if size == 0:
        return 0.0
    numerator = sum(lhs[index] * rhs[index] for index in range(size))
    lhs_norm = math.sqrt(sum(lhs[index] * lhs[index] for index in range(size)))
    rhs_norm = math.sqrt(sum(rhs[index] * rhs[index] for index in range(size)))
    if lhs_norm == 0 or rhs_norm == 0:
        return 0.0
    return numerator / (lhs_norm * rhs_norm)


def _inline_multivectors_enabled() -> bool:
    raw = os.getenv("RAG_MULTIVECTOR_INLINE", "false")
    return raw.lower() in {"true", "1", "yes"}


def _lightweight_multivectors_allowed() -> bool:
    raw = os.getenv("RAG_ALLOW_LIGHTWEIGHT_MULTIVECTOR", "false")
    return raw.lower() in {"true", "1", "yes"}


def _is_visual_multivector_chunk(chunk: Chunk) -> bool:
    metadata = chunk.metadata or {}
    return (
        chunk.modality in {"image", "document"}
        or metadata.get("chunk_kind")
        in {"rendered_page_image", "embedded_image", "media_object", "document_attachment"}
        or metadata.get("chunk_strategy")
        in {"rendered_page_image", "embedded_image", "media_object", "page_attachment"}
    )


def _chunk_media_path(chunk: Chunk) -> Path | None:
    raw = chunk.media_uri or (chunk.metadata or {}).get("media_uri")
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw)
    return path if path.exists() else None


def _is_image_path(path: Path, mime_type: str | None = None) -> bool:
    mime = mime_type or _guess_media_mime(path)
    return mime.startswith("image/")


def _guess_media_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".pdf": "application/pdf",
    }.get(suffix, "application/octet-stream")


def _data_url(path: Path, mime_type: str) -> str:
    import base64

    return f"data:{mime_type};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _extract_provider_vectors(body: object) -> list[list[float]]:
    if not isinstance(body, dict):
        raise ModelGatewayError(f"multivector provider returned non-object body: {body}")
    candidates = [
        body.get("vectors"),
        body.get("embeddings"),
        body.get("multi_vector"),
    ]
    output = body.get("output")
    if isinstance(output, dict):
        candidates.extend(
            [
                output.get("vectors"),
                output.get("embeddings"),
                output.get("multi_vector"),
            ]
        )
    data = body.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            candidates.extend(
                [
                    first.get("embedding"),
                    first.get("vectors"),
                    first.get("multi_vector"),
                ]
            )
    for candidate in candidates:
        vectors = _coerce_provider_vector_candidate(candidate)
        if vectors:
            return vectors
    raise ModelGatewayError(f"multivector provider returned no vectors: {body}")


def _coerce_provider_vector_candidate(candidate: object) -> list[list[float]]:
    if not isinstance(candidate, list) or not candidate:
        return []
    if all(isinstance(value, (int, float)) for value in candidate):
        return [[float(value) for value in candidate]]
    return _coerce_vectors(candidate)


def _tensor_to_vectors(tensor: object) -> list[list[float]]:
    value = tensor
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        raw = value.tolist()
    else:
        raw = value
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        if raw and raw[0] and isinstance(raw[0][0], list):
            raw = raw[0]
        return _coerce_vectors(raw)
    return []


def _model_device(model: object) -> object:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device  # type: ignore[attr-defined]
    except Exception:
        return "cpu"


def _torch_dtype(torch_module: object, raw: str) -> object | None:
    value = raw.lower().strip()
    if value in {"", "auto", "none"}:
        return None
    if value in {"bf16", "bfloat16"}:
        return getattr(torch_module, "bfloat16")
    if value in {"fp16", "float16", "half"}:
        return getattr(torch_module, "float16")
    if value in {"fp32", "float32"}:
        return getattr(torch_module, "float32")
    return None


def _int_setting(env_name: str, configured: object, default: int) -> int:
    raw = os.getenv(env_name)
    value = raw if raw is not None else configured
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default
