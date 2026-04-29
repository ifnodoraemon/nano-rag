from __future__ import annotations

import os
import re
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.core.exceptions import ConfigurationError
from app.core.tracing import FeedbackStore, TraceStore, TracingManager
from app.generation.answer_formatter import AnswerFormatter
from app.generation.prompt_builder import PromptBuilder
from app.generation.service import GenerationService
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.semantic_chunker import SemanticChunker, SemanticChunkerConfig
from app.model_client.document_parser import DocumentParserClient
from app.model_client.embeddings import EmbeddingClient
from app.model_client.generation import GenerationClient
from app.model_client.rerank import RerankClient
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.query_rewriter import QueryRewriter, QueryRewriterConfig
from app.utils.text import parse_bool_env
from app.vectorstore.repository import (
    InMemoryVectorRepository,
    MilvusVectorRepository,
    VectorRepository,
)
from app.wiki.compiler import WikiCompiler
from app.wiki.search import WikiSearcher


def _render_env(raw: str) -> str:
    def replace(match: re.Match[str]) -> str:
        expression = match.group(1)
        if ":-" in expression:
            key, default = expression.split(":-", 1)
            return os.getenv(key, default)
        return os.getenv(expression, match.group(0))

    return re.sub(r"\$\{([^}]+)\}", replace, raw)


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(_render_env(path.read_text())) or {}


INSECURE_DEFAULT_KEYS = frozenset({"change-me", "sk-xxx", "your-api-key", ""})


@dataclass
class AppConfig:
    config_dir: Path
    settings: dict[str, Any]
    models: dict[str, Any]
    prompts: dict[str, Any]

    @property
    def gateway_base_url(self) -> str:
        return self.models["model_gateway"]["base_url"]

    @property
    def gateway_api_key(self) -> str:
        explicit = os.getenv("MODEL_GATEWAY_API_KEY")
        if explicit:
            return explicit
        key = self.models["model_gateway"]["api_key"]
        if key in INSECURE_DEFAULT_KEYS:
            raise ConfigurationError(
                "MODEL_GATEWAY_API_KEY must be set for production mode. "
                "Configure a real provider key explicitly."
            )
        return str(key)

    def gateway_for(self, capability: str, validate: bool = True) -> dict[str, str]:
        env_prefix_map = {
            "embedding": "EMBEDDING",
            "generation": "GENERATION",
            "rerank": "RERANK",
        }
        env_prefix = env_prefix_map.get(capability, capability.upper())
        section = self.models.get(capability, {})

        base_url = (
            os.getenv(f"{env_prefix}_API_BASE_URL")
            or section.get("base_url")
        )
        api_key = (
            os.getenv(f"{env_prefix}_API_KEY")
            or section.get("api_key")
        )
        if not validate:
            return {"base_url": str(base_url or ""), "api_key": str(api_key or "")}
        if not base_url:
            raise ConfigurationError(
                f"{capability.upper()}_API_BASE_URL must be configured explicitly."
            )
        if api_key in INSECURE_DEFAULT_KEYS:
            raise ConfigurationError(
                f"{capability.upper()}_API_KEY must be set for production mode. "
                "No alternate backend key will be used."
            )
        return {"base_url": str(base_url), "api_key": str(api_key)}

    @property
    def gateway_models_probe_paths(self) -> tuple[str, ...]:
        return ("/v1/models", "/models")

    @property
    def gateway_mode(self) -> str:
        mode = os.getenv("MODEL_GATEWAY_MODE", "live").lower()
        if mode == "mock":
            raise ConfigurationError(
                "MODEL_GATEWAY_MODE=mock is not supported in the real-data runtime."
            )
        return mode

    @property
    def rerank_enabled(self) -> bool:
        if parse_bool_env(os.getenv("DISABLE_RERANK")):
            return False
        alias = (
            str(self.models.get("rerank", {}).get("default_alias", "")).strip().lower()
        )
        return alias not in {"", "disabled", "none", "null"}

    @property
    def parsed_dir(self) -> Path:
        return Path(
            os.getenv("PARSED_OUTPUT_DIR", self.config_dir.parent / "data" / "parsed")
        )

    @property
    def langfuse_otel_endpoint(self) -> str:
        return os.getenv("LANGFUSE_OTEL_ENDPOINT", "").strip()

    @property
    def langfuse_ui_endpoint(self) -> str:
        return os.getenv("LANGFUSE_UI_ENDPOINT", "").strip()

    @property
    def langfuse_public_key(self) -> str:
        return os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()

    @property
    def langfuse_secret_key(self) -> str:
        return os.getenv("LANGFUSE_SECRET_KEY", "").strip()

    @property
    def langfuse_otel_headers(self) -> dict[str, str]:
        if not self.langfuse_public_key or not self.langfuse_secret_key:
            return {}
        token = base64.b64encode(
            f"{self.langfuse_public_key}:{self.langfuse_secret_key}".encode("utf-8")
        ).decode("ascii")
        return {"Authorization": f"Basic {token}"}

    @property
    def document_parser_configured(self) -> dict[str, object]:
        section = self.models.get("document_parser", {})
        enabled_raw = os.getenv("DOCUMENT_PARSER_ENABLED")
        if enabled_raw is None:
            enabled_value = section.get("enabled", True)
            enabled = enabled_value if isinstance(enabled_value, bool) else parse_bool_env(str(enabled_value))
        else:
            enabled = parse_bool_env(enabled_raw)
        provider = (
            os.getenv("DOCUMENT_PARSER_PROVIDER")
            or section.get("provider")
            or "gemini"
        )
        model = (
            os.getenv("DOCUMENT_PARSER_MODEL")
            or section.get("default_alias")
            or ""
        )
        base_url = (
            os.getenv("DOCUMENT_PARSER_API_BASE_URL")
            or section.get("base_url")
            or ""
        )
        api_key = os.getenv("DOCUMENT_PARSER_API_KEY") or section.get("api_key") or ""
        missing: list[str] = []
        if enabled and not str(base_url).strip():
            missing.append("DOCUMENT_PARSER_API_BASE_URL")
        if enabled and not str(model).strip():
            missing.append("DOCUMENT_PARSER_MODEL")
        if enabled and str(api_key).strip() in INSECURE_DEFAULT_KEYS:
            missing.append("DOCUMENT_PARSER_API_KEY")
        return {
            "enabled": enabled,
            "provider": str(provider),
            "model": str(model),
            "base_url": str(base_url) if base_url else None,
            "configured": enabled and not missing,
            "missing": missing,
        }

    @property
    def trace_store_dir(self) -> Path:
        return Path(
            os.getenv(
                "TRACE_STORE_DIR",
                self.config_dir.parent / "data" / "reports" / "traces",
            )
        )

    @property
    def feedback_store_dir(self) -> Path:
        return Path(
            os.getenv(
                "FEEDBACK_STORE_DIR",
                self.config_dir.parent / "data" / "reports" / "feedback",
            )
        )

    @property
    def wiki_dir(self) -> Path:
        return Path(
            os.getenv("WIKI_OUTPUT_DIR", self.config_dir.parent / "data" / "wiki")
        )

    @property
    def upload_dir(self) -> Path:
        return Path(
            os.getenv("UPLOAD_OUTPUT_DIR", self.config_dir.parent / "data" / "uploads")
        )

    @property
    def business_api_keys(self) -> set[str]:
        raw = os.getenv("RAG_API_KEYS", "")
        return {item.strip() for item in raw.split(",") if item.strip()}

    @property
    def wiki_enabled(self) -> bool:
        return parse_bool_env(os.getenv("RAG_WIKI_ENABLED"))

    @property
    def hybrid_search_enabled(self) -> bool:
        raw = os.getenv("RAG_HYBRID_SEARCH_ENABLED")
        if raw is not None:
            return parse_bool_env(raw)
        return bool(self.settings.get("hybrid_search", {}).get("enabled", False))

    @property
    def semantic_chunker_enabled(self) -> bool:
        return parse_bool_env(os.getenv("RAG_SEMANTIC_CHUNKER_ENABLED"))

    @property
    def query_rewrite_enabled(self) -> bool:
        query_rewriter_config = QueryRewriterConfig.from_env()
        return (
            query_rewriter_config.enable_rewrite
            or query_rewriter_config.enable_multi_query
            or query_rewriter_config.enable_hyde
        )

    @property
    def diagnosis_enabled(self) -> bool:
        return parse_bool_env(os.getenv("RAG_DIAGNOSIS_ENABLED"))

    @property
    def eval_enabled(self) -> bool:
        return parse_bool_env(os.getenv("RAG_EVAL_ENABLED"))

    @property
    def benchmark_enabled(self) -> bool:
        return self.eval_enabled and self.diagnosis_enabled


def load_config() -> AppConfig:
    config_dir = Path(
        os.getenv("APP_CONFIG_DIR", Path(__file__).resolve().parents[2] / "configs")
    )
    return AppConfig(
        config_dir=config_dir,
        settings=_load_yaml(config_dir / "settings.yaml"),
        models=_load_yaml(config_dir / "models.yaml"),
        prompts=_load_yaml(config_dir / "prompts.yaml"),
    )


def build_repository(config: AppConfig) -> VectorRepository:
    backend = os.getenv("VECTORSTORE_BACKEND", "milvus").lower()
    if backend == "milvus":
        return MilvusVectorRepository.from_config(config)
    if backend == "memory":
        return InMemoryVectorRepository()
    raise ConfigurationError(
        f"Unsupported VECTORSTORE_BACKEND={backend!r}; expected 'milvus' or 'memory'."
    )


@dataclass
class AppContainer:
    config: AppConfig
    repository: VectorRepository
    embedding_client: EmbeddingClient
    rerank_client: RerankClient
    generation_client: GenerationClient
    document_parser: DocumentParserClient
    ingestion_pipeline: IngestionPipeline
    retrieval_pipeline: RetrievalPipeline
    chat_pipeline: GenerationService
    ragas_runner: object | None
    trace_store: TraceStore
    tracing_manager: TracingManager
    diagnosis_service: object | None
    feedback_store: FeedbackStore
    query_rewriter: QueryRewriter | None = None
    semantic_chunker: SemanticChunker | None = None
    hybrid_retriever: HybridRetriever | None = None
    wiki_compiler: WikiCompiler | None = None
    wiki_searcher: WikiSearcher | None = None

    async def close(self) -> None:
        await self.embedding_client.close()
        await self.rerank_client.close()
        await self.generation_client.close()
        await self.document_parser.close()
        if hasattr(self.repository, "close"):
            await self.repository.close()

    @classmethod
    def from_env(cls) -> "AppContainer":
        config = load_config()
        repository = build_repository(config)
        embedding_client = EmbeddingClient(config)
        rerank_client = RerankClient(config)
        generation_client = GenerationClient(config)
        document_parser = DocumentParserClient(config)
        trace_store = TraceStore(persist_dir=config.trace_store_dir)
        feedback_store = FeedbackStore(persist_dir=config.feedback_store_dir)
        tracing_manager = TracingManager(
            "nano-rag",
            config.langfuse_otel_endpoint,
            headers=config.langfuse_otel_headers,
        )
        wiki_compiler = WikiCompiler(config.wiki_dir) if config.wiki_enabled else None
        wiki_searcher = WikiSearcher(config.wiki_dir) if config.wiki_enabled else None
        query_rewriter = None
        if config.query_rewrite_enabled:
            query_rewriter_config = QueryRewriterConfig.from_env()
            query_rewriter = QueryRewriter(
                generation_client=generation_client,
                config=query_rewriter_config,
            )
        hybrid_retriever = None
        if config.hybrid_search_enabled:
            hybrid_retriever = HybridRetriever(
                repository=repository,
                embedding_client=embedding_client,
            )
            hybrid_retriever.bootstrap_from_parsed_dir(config.parsed_dir)
        semantic_chunker = None
        if config.semantic_chunker_enabled:
            semantic_chunker_config = SemanticChunkerConfig.from_env()
            semantic_chunker = SemanticChunker(config=semantic_chunker_config)
        ragas_runner = None
        if config.eval_enabled:
            from app.eval.ragas_runner import RagasRunner

            ragas_runner = RagasRunner(generation_client=generation_client)
        diagnosis_service = None
        if config.diagnosis_enabled:
            from app.diagnostics.service import DiagnosisService

            diagnosis_service = DiagnosisService(generation_client=generation_client)
        retrieval_pipeline = RetrievalPipeline(
            config,
            repository,
            embedding_client,
            rerank_client,
            trace_store,
            tracing_manager,
            query_rewriter=query_rewriter,
            hybrid_retriever=hybrid_retriever,
            wiki_searcher=wiki_searcher,
        )
        chat_pipeline = GenerationService(
            config=config,
            retrieval_pipeline=retrieval_pipeline,
            generation_client=generation_client,
            prompt_builder=PromptBuilder(config.prompts),
            answer_formatter=AnswerFormatter(),
            trace_store=trace_store,
            tracing_manager=tracing_manager,
        )
        return cls(
            config=config,
            repository=repository,
            embedding_client=embedding_client,
            rerank_client=rerank_client,
            generation_client=generation_client,
            document_parser=document_parser,
            ingestion_pipeline=IngestionPipeline(
                config,
                repository,
                embedding_client,
                tracing_manager,
                semantic_chunker,
                document_parser=document_parser,
                hybrid_retriever=hybrid_retriever,
                wiki_compiler=wiki_compiler,
                wiki_searcher=wiki_searcher,
            ),
            retrieval_pipeline=retrieval_pipeline,
            chat_pipeline=chat_pipeline,
            ragas_runner=ragas_runner,
            trace_store=trace_store,
            tracing_manager=tracing_manager,
            diagnosis_service=diagnosis_service,
            feedback_store=feedback_store,
            query_rewriter=query_rewriter,
            semantic_chunker=semantic_chunker,
            hybrid_retriever=hybrid_retriever,
            wiki_compiler=wiki_compiler,
            wiki_searcher=wiki_searcher,
        )
