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
    app.conf.worker_prefetch_multiplier = int(
        os.getenv("RAG_WORKER_PREFETCH_MULTIPLIER", "1")
    )
    app.conf.worker_max_tasks_per_child = int(
        os.getenv("RAG_WORKER_MAX_TASKS_PER_CHILD", "20")
    )
    app.conf.task_acks_late = True
    app.conf.task_reject_on_worker_lost = True
    app.conf.task_track_started = True
    app.conf.broker_connection_retry_on_startup = True
    app.conf.task_soft_time_limit = int(os.getenv("RAG_INGEST_TASK_SOFT_TIME_LIMIT", "1800"))
    app.conf.task_time_limit = int(os.getenv("RAG_INGEST_TASK_TIME_LIMIT", "2100"))
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

    @celery_app.task(
        name="nano_rag.ingest_paths",
        queue="ingest",
        autoretry_for=(Exception,),
        retry_backoff=True,
        retry_jitter=True,
        retry_kwargs={
            "max_retries": int(os.getenv("RAG_INGEST_TASK_MAX_RETRIES", "2")),
        },
    )
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
