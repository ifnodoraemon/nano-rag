from __future__ import annotations

import logging
import os
from collections.abc import Sequence

from fastapi import BackgroundTasks

logger = logging.getLogger(__name__)


async def run_ingest_paths(
    container,
    job_id: str,
    paths: Sequence[str],
    kb_id: str,
    source_path_overrides: dict[str, str] | None = None,
) -> None:
    documents = 0
    chunks = 0
    try:
        container.ingest_job_store.mark_running(job_id, stage="ingesting")
        for index, path in enumerate(paths):
            if index:
                # Per-path liveness marker: a long batch is not "stale".
                container.ingest_job_store.heartbeat(
                    job_id, stage=f"ingesting {index + 1}/{len(paths)}"
                )
            response = await container.ingestion_pipeline.run(
                path,
                kb_id=kb_id,
                source_path_overrides=source_path_overrides,
            )
            documents += response.documents
            chunks += response.chunks
        container.ingest_job_store.mark_completed(
            job_id,
            documents=documents,
            chunks=chunks,
        )
    except Exception as exc:
        container.ingest_job_store.mark_failed(
            job_id,
            stage="failed",
            error=str(exc),
        )
        raise
    except BaseException as exc:
        # Cancellation (client disconnect, worker abort, asyncio.run teardown)
        # must not leave the record RUNNING with no owner.
        container.ingest_job_store.mark_failed(
            job_id,
            stage="failed",
            error=f"ingest job was cancelled or aborted: {type(exc).__name__}",
        )
        raise


def submit_ingest_paths(
    *,
    background_tasks: BackgroundTasks,
    container,
    job_id: str,
    paths: Sequence[str],
    kb_id: str,
    source_path_overrides: dict[str, str] | None = None,
) -> None:
    if os.getenv("RAG_INGEST_EXECUTOR", "background").lower() == "celery":
        from app.ingestion.tasks import ingest_paths_task

        ingest_paths_task.delay(job_id, list(paths), kb_id, source_path_overrides or {})
        return

    background_tasks.add_task(
        _run_ingest_paths_background,
        container,
        job_id,
        list(paths),
        kb_id,
        source_path_overrides,
    )


async def _run_ingest_paths_background(
    container,
    job_id: str,
    paths: Sequence[str],
    kb_id: str,
    source_path_overrides: dict[str, str] | None = None,
) -> None:
    """Background-task wrapper that contains the failure.

    ``run_ingest_paths`` re-raises because Celery needs the exception for its
    autoretry policy. A FastAPI BackgroundTask runs after the response is sent,
    so re-raising there only escapes into the ASGI server as an unhandled
    traceback — the job record already carries the failure, so log and stop.
    """
    try:
        await run_ingest_paths(
            container,
            job_id,
            paths,
            kb_id,
            source_path_overrides,
        )
    except Exception:  # noqa: BLE001 - already recorded on the job
        logger.exception("background ingest job %s failed", job_id)
