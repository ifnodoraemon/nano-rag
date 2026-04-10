from __future__ import annotations

import asyncio
import logging
import os
import time

from pymilvus import MilvusClient
from pymilvus.exceptions import MilvusException

LOGGER = logging.getLogger(__name__)


def _create_milvus_client_sync() -> MilvusClient:
    uri = os.getenv("MILVUS_URI", "http://localhost:19530")
    attempts = max(1, int(os.getenv("MILVUS_CONNECT_MAX_ATTEMPTS", "30")))
    retry_seconds = max(0.1, float(os.getenv("MILVUS_CONNECT_RETRY_SECONDS", "2")))
    last_error: MilvusException | None = None

    for attempt in range(1, attempts + 1):
        try:
            return MilvusClient(uri=uri)
        except MilvusException as exc:
            last_error = exc
            if attempt == attempts:
                break
            LOGGER.warning(
                "Milvus is not ready yet; retrying connection (%s/%s) to %s in %.1fs",
                attempt,
                attempts,
                uri,
                retry_seconds,
            )
            time.sleep(retry_seconds)

    if last_error is not None:
        raise last_error
    raise RuntimeError("unexpected Milvus initialization failure")


def create_milvus_client() -> MilvusClient:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_create_milvus_client_sync).result()
    return _create_milvus_client_sync()
