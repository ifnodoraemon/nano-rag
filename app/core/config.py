from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.eval.ragas_runner import RagasRunner
from app.core.tracing import TraceStore, TracingManager
from app.generation.answer_formatter import AnswerFormatter
from app.generation.prompt_builder import PromptBuilder
from app.generation.service import GenerationService
from app.ingestion.pipeline import IngestionPipeline
from app.model_client.embeddings import EmbeddingClient
from app.model_client.generation import GenerationClient
from app.model_client.rerank import RerankClient
from app.retrieval.pipeline import RetrievalPipeline
from app.vectorstore.repository import InMemoryVectorRepository, MilvusVectorRepository, VectorRepository


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
        return self.models["model_gateway"]["api_key"]

    @property
    def gateway_mode(self) -> str:
        return os.getenv("MODEL_GATEWAY_MODE", "mock").lower()

    @property
    def parsed_dir(self) -> Path:
        return Path(os.getenv("PARSED_OUTPUT_DIR", self.config_dir.parent / "data" / "parsed"))

    @property
    def phoenix_collector_endpoint(self) -> str:
        return os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix:4317")

    @property
    def phoenix_ui_endpoint(self) -> str:
        return os.getenv("PHOENIX_UI_ENDPOINT", "http://phoenix:6006")


def load_config() -> AppConfig:
    config_dir = Path(os.getenv("APP_CONFIG_DIR", Path(__file__).resolve().parents[2] / "configs"))
    return AppConfig(
        config_dir=config_dir,
        settings=_load_yaml(config_dir / "settings.yaml"),
        models=_load_yaml(config_dir / "models.yaml"),
        prompts=_load_yaml(config_dir / "prompts.yaml"),
    )


def build_repository(config: AppConfig) -> VectorRepository:
    if os.getenv("VECTORSTORE_BACKEND", "memory").lower() == "milvus":
        return MilvusVectorRepository.from_config(config)
    return InMemoryVectorRepository()


@dataclass
class AppContainer:
    config: AppConfig
    repository: VectorRepository
    embedding_client: EmbeddingClient
    rerank_client: RerankClient
    generation_client: GenerationClient
    ingestion_pipeline: IngestionPipeline
    retrieval_pipeline: RetrievalPipeline
    chat_pipeline: GenerationService
    ragas_runner: RagasRunner
    trace_store: TraceStore
    tracing_manager: TracingManager

    @classmethod
    def from_env(cls) -> "AppContainer":
        config = load_config()
        repository = build_repository(config)
        embedding_client = EmbeddingClient(config)
        rerank_client = RerankClient(config)
        generation_client = GenerationClient(config)
        trace_store = TraceStore()
        tracing_manager = TracingManager("nano-rag", config.phoenix_collector_endpoint)
        retrieval_pipeline = RetrievalPipeline(
            config,
            repository,
            embedding_client,
            rerank_client,
            trace_store,
            tracing_manager,
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
            ingestion_pipeline=IngestionPipeline(config, repository, embedding_client, tracing_manager),
            retrieval_pipeline=retrieval_pipeline,
            chat_pipeline=chat_pipeline,
            ragas_runner=RagasRunner(),
            trace_store=trace_store,
            tracing_manager=tracing_manager,
        )
