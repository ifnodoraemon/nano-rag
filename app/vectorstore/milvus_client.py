from __future__ import annotations

import os

from pymilvus import MilvusClient


def create_milvus_client() -> MilvusClient:
    return MilvusClient(uri=os.getenv("MILVUS_URI", "http://localhost:19530"))
