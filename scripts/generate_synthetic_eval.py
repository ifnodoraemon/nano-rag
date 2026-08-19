#!/usr/bin/env python3
import asyncio
import json
import logging
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import AppContainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _embedding_dimension(container: AppContainer) -> int:
    # Source of truth is the vector store config (matches the collection dim);
    # never hardcode it here or a model/dimension change breaks the probe vector.
    dimension = getattr(container.repository, "dimension", None)
    if isinstance(dimension, int) and dimension > 0:
        return dimension
    configured = container.config.models.get("embedding", {}).get("dimension")
    if isinstance(configured, int) and configured > 0:
        return configured
    raise SystemExit("Unable to determine embedding dimension from container config.")


async def generate_synthetic_data(container: AppContainer, limit: int = 10) -> int:
    """
    Scrapes chunks from the vector store and asks the LLM to generate Q&A pairs
    to create a synthetic dataset for robust evaluation. Returns the record count.
    """
    logger.info("Initializing synthetic dataset generation...")
    vector_repo = container.repository
    generation_client = container.generation_client

    dummy_vector = [random.random() for _ in range(_embedding_dimension(container))]

    hits = vector_repo.search(
        vector=dummy_vector,
        top_k=limit * 2,
    )

    if not hits:
        logger.error("No chunks found in the database. Please ingest documents first.")
        return 0

    logger.info(f"Retrieved {len(hits)} chunks for generation.")

    synthetic_records = []

    prompt = """
    你是一个专业的数据集构建专家。请阅读以下文本片段，并基于它生成 1 个高质量的 RAG 问答测试对。

    要求：
    1. 问题 (query) 必须是一个自然语言的提问，难度可以是事实型或推理性。
    2. 答案 (expected_answer) 必须完全依据提供的文本片段，且事实准确。
    3. 输出必须是合法的 JSON 格式。

    输出 JSON 格式要求：
    {{
        "query": "生成的问题",
        "expected_answer": "生成的答案"
    }}

    文本片段：
    {text}
    """

    for i, hit in enumerate(hits[:limit]):
        text = hit.chunk.text
        messages = [
            {"role": "system", "content": "You are a helpful dataset generator."},
            {"role": "user", "content": prompt.format(text=text)}
        ]

        try:
            result = await generation_client.generate(messages)
            content = result["content"]
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = content[json_start:json_end]
                qa_pair = json.loads(json_str)

                record = {
                    "sample_id": f"synthetic-{i}",
                    "kb_id": "default",
                    "query": qa_pair["query"],
                    "reference_answer": qa_pair["expected_answer"],
                    "reference_contexts": [text[:200] + "..."]
                }
                synthetic_records.append(record)
                logger.info(f"Generated record {i+1}/{limit}")
        except Exception as e:
            logger.warning(f"Failed to generate QA for chunk {i}: {e}")

    output_path = ROOT / "data" / "eval" / "rag_synthetic_dataset.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in synthetic_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Successfully generated {len(synthetic_records)} records to {output_path}")
    return len(synthetic_records)


if __name__ == "__main__":
    container = AppContainer.from_env()
    produced = asyncio.run(generate_synthetic_data(container, limit=20))
    if produced == 0:
        logger.error("No synthetic records were produced; refusing to report success.")
        sys.exit(1)
