from __future__ import annotations

import os
from collections.abc import Sequence

from fastapi import BackgroundTasks


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
        for path in paths:
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
        run_ingest_paths,
        container,
        job_id,
        list(paths),
        kb_id,
        source_path_overrides,
    )
