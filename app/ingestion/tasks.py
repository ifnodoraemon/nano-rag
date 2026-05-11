from __future__ import annotations

import asyncio
import os

from app.core.config import AppContainer
from app.core.logging import configure_logging
from app.ingestion.executor import run_ingest_paths

try:
    from celery import Celery
except ImportError:  # pragma: no cover - Celery is optional outside worker runtime.
    Celery = None


def _build_celery_app():
    if Celery is None:
        raise RuntimeError("celery is not installed")
    broker_url = os.getenv("RAG_BROKER_URL", "redis://redis:6379/0")
    backend_url = os.getenv("RAG_RESULT_BACKEND", broker_url)
    app = Celery("nano_rag", broker=broker_url, backend=backend_url)
    app.conf.task_default_queue = "ingest"
    app.conf.worker_prefetch_multiplier = 1
    app.conf.task_acks_late = True
    return app


celery_app = _build_celery_app() if Celery is not None else None


async def _run_job(
    job_id: str,
    paths: list[str],
    kb_id: str,
    source_path_overrides: dict[str, str],
) -> None:
    configure_logging()
    container = AppContainer.from_env()
    try:
        await run_ingest_paths(
            container,
            job_id,
            paths,
            kb_id,
            source_path_overrides,
        )
    finally:
        await container.close()


if celery_app is not None:

    @celery_app.task(name="nano_rag.ingest_paths", queue="ingest")
    def ingest_paths_task(
        job_id: str,
        paths: list[str],
        kb_id: str,
        source_path_overrides: dict[str, str] | None = None,
    ) -> None:
        asyncio.run(_run_job(job_id, paths, kb_id, source_path_overrides or {}))

else:

    class _MissingCeleryTask:
        def delay(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise RuntimeError("celery is not installed")

    ingest_paths_task = _MissingCeleryTask()
