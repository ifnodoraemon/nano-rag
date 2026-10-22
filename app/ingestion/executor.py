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
    force: bool = False,
) -> None:
    documents = 0
    chunks = 0
    unchanged = 0
    prompt_tokens = 0
    completion_tokens = 0
    try:
        container.ingest_job_store.mark_running(job_id, stage="ingesting")
        for index, path in enumerate(paths):
            # Heartbeat on EVERY path, including the first: a single large
            # document used to run for its whole duration without ever refreshing
            # the marker, so a slow but healthy ingest was reaped as abandoned
            # once it exceeded the stale threshold.
            container.ingest_job_store.heartbeat(
                job_id, stage=f"ingesting {index + 1}/{len(paths)}"
            )
            response = await container.ingestion_pipeline.run(
                path,
                kb_id=kb_id,
                source_path_overrides=source_path_overrides,
                force=force,
            )
            documents += response.documents
            chunks += response.chunks
            unchanged += len(response.unchanged_doc_ids)
            usage = response.usage or {}
            prompt_tokens += int(usage.get("prompt_tokens") or 0)
            completion_tokens += int(usage.get("completion_tokens") or 0)
        _record_completion(
            container,
            job_id,
            documents=documents,
            chunks=chunks,
            unchanged_documents=unchanged,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    except Exception as exc:
        _record_failure(container, job_id, stage="failed", error=str(exc))
        raise
    except BaseException as exc:
        # Cancellation (client disconnect, worker abort, asyncio.run teardown)
        # must not leave the record RUNNING with no owner.
        _record_failure(
            container,
            job_id,
            stage="failed",
            error=f"ingest job was cancelled or aborted: {type(exc).__name__}",
        )
        raise


def _record_completion(
    container,
    job_id: str,
    *,
    documents: int,
    chunks: int,
    unchanged_documents: int,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    """Write the terminal record, without ever masking the caller's outcome."""
    try:
        container.ingest_job_store.mark_completed(
            job_id,
            documents=documents,
            chunks=chunks,
            unchanged_documents=unchanged_documents,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    except Exception:  # noqa: BLE001 - the work is already done and durably stored
        logger.exception(
            "ingest job %s finished but its completion record could not be written",
            job_id,
        )


def _record_failure(container, job_id: str, *, stage: str, error: str) -> None:
    """Record a failed job without replacing the exception being propagated.

    ``mark_failed`` used to be called inline inside the ``except`` block. When the
    store itself was unwritable - or the record had been purged, so ``_require``
    raised ``KeyError`` - that new exception REPLACED the real cause, and the
    operator saw "ingest job not found" instead of the provider failure. The
    original error must always win; the store problem is logged instead.
    """
    try:
        container.ingest_job_store.mark_failed(job_id, stage=stage, error=error)
    except Exception:  # noqa: BLE001 - deliberately never re-raised
        logger.exception(
            "ingest job %s failed (%s) and its failure record could not be "
            "written; the record may still show RUNNING",
            job_id,
            error,
        )


def submit_ingest_paths(
    *,
    background_tasks: BackgroundTasks,
    container,
    job_id: str,
    paths: Sequence[str],
    kb_id: str,
    source_path_overrides: dict[str, str] | None = None,
    force: bool = False,
) -> None:
    if os.getenv("RAG_INGEST_EXECUTOR", "background").lower() == "celery":
        from app.ingestion.tasks import ingest_paths_task

        try:
            ingest_paths_task.delay(
                job_id, list(paths), kb_id, source_path_overrides or {}, force
            )
        except Exception as exc:
            # The record was created before submitting. If the broker is down the
            # job would sit QUEUED forever: nothing reaps QUEUED work and no client
            # can distinguish "waiting" from "never scheduled".
            _record_failure(
                container,
                job_id,
                stage="submit-failed",
                error=f"could not enqueue the ingest job: {exc}",
            )
            raise
        return

    background_tasks.add_task(
        _run_ingest_paths_background,
        container,
        job_id,
        list(paths),
        kb_id,
        source_path_overrides,
        force,
    )


async def _run_ingest_paths_background(
    container,
    job_id: str,
    paths: Sequence[str],
    kb_id: str,
    source_path_overrides: dict[str, str] | None = None,
    force: bool = False,
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
            force=force,
        )
    except Exception:  # noqa: BLE001 - already recorded on the job
        logger.exception("background ingest job %s failed", job_id)
