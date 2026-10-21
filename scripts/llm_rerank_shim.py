#!/usr/bin/env python3
"""A rerank endpoint backed by a chat model, for A/B experiments.

The project's rerank stage is a standard cross-encoder contract:

    POST <RERANK_API_BASE_URL><RERANK_API_PATH>
      -> {"model": ..., "query": str, "documents": [str], "top_n": int}
    <- {"results": [{"index": int, "relevance_score": float}, ...]}

The bundled model gateway has no rerank endpoint, so this shim answers that
contract using the chat model: it asks the model to score every candidate in one
call and returns the same shape. Its purpose is to measure *whether reranking
helps this corpus*, exercising the real RerankClient and the real discovery seam
- NOT to compare a cross-encoder against BM25. A cross-encoder scores
(query, document) pairs directly and is usually both better calibrated and far
cheaper; treat the absolute numbers here as an upper bound on latency and a
lower bound on quality.

    python scripts/llm_rerank_shim.py --port 8099

Then point the app at it:

    RERANK_MODEL_ALIAS=llm-rerank-shim
    RERANK_API_BASE_URL=http://host.docker.internal:8099   # or http://localhost:8099
    RERANK_API_PATH=/reranks
    DISABLE_RERANK=false
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from typing import Any

import httpx

logger = logging.getLogger("llm_rerank_shim")

from pydantic import BaseModel


class RerankRequest(BaseModel):
    model: str | None = None
    query: str
    documents: list[str]
    top_n: int | None = None


DEFAULT_GATEWAY = "http://localhost:8080/v1"
DEFAULT_MODEL = "deepseek-v4.1-flash"

PROMPT = """You are a relevance judge for a search engine.

For each candidate passage, score how well it answers the user's query on a
scale from 0 to 100:
- 100 = directly and completely answers the query
- 50  = related topic but does not answer it
- 0   = unrelated

Return ONLY a JSON object of the form {"scores": [<int>, ...]} with exactly one
score per candidate, in the same order as the candidates. No prose, no markdown.

Query: __QUERY__

Candidates:
__CANDIDATES__
"""


def _render_prompt(query: str, documents: list[str]) -> str:
    """Fill the template without str.format().

    The template contains a literal JSON example ({"scores": [...]}); .format()
    reads those braces as placeholders and raises KeyError('"scores"').
    """
    return PROMPT.replace("__QUERY__", query).replace(
        "__CANDIDATES__", _candidates_block(documents)
    )


def _candidates_block(documents: list[str]) -> str:
    lines = []
    for index, document in enumerate(documents):
        text = " ".join(str(document).split())[:600]
        lines.append(f"[{index}] {text}")
    return "\n".join(lines)


def _extract_scores(content: str, expected: int) -> list[float]:
    """Parse the model's scores, tolerating a fenced or chatty reply."""
    text = str(content or "")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")
    payload = json.loads(match.group(0))
    raw = payload.get("scores")
    if not isinstance(raw, list):
        # A non-list payload is a shape violation, not a value error.
        raise TypeError(f"reply has no 'scores' array: {payload!r}")
    scores = []
    for value in raw[:expected]:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise TypeError(f"non-numeric score {value!r}")
        try:
            scores.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"non-numeric score {value!r}") from exc
    if len(scores) != expected:
        raise ValueError(f"expected {expected} scores, got {len(scores)}")
    return scores


def _score(
    gateway: str,
    api_key: str,
    model: str,
    query: str,
    documents: list[str],
    *,
    attempts: int = 5,
) -> list[float]:
    prompt = _render_prompt(query, documents)
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 512,
    }
    # Upstream gateways occasionally return a transient 404/5xx (routing hiccup,
    # rate limit). This shim is an offline experiment tool, so retry here rather
    # than abort a multi-minute benchmark on one blip. The application itself
    # still fails loudly on a rerank error - that behaviour is deliberate.
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with httpx.Client(timeout=120) as client:
                response = client.post(
                    f"{gateway.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=body,
                )
                response.raise_for_status()
                data = response.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            return _extract_scores(content, len(documents))
        except Exception as exc:  # noqa: BLE001 - retried, then re-raised
            last_error = exc
            logger.warning("scoring attempt %d/%d failed: %s", attempt, attempts, exc)
            if attempt < attempts:
                # Upstream gateways blip for a few seconds; a short backoff just
                # spends every retry inside the bad window.
                time.sleep(min(20.0, 3.0 * (2 ** (attempt - 1))))
    raise RuntimeError(f"scoring failed after {attempts} attempts: {last_error}")


def build_app(gateway: str, api_key: str, model: str):
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="llm-rerank-shim")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "backend": "chat-model"}

    @app.post("/reranks")
    async def reranks(request: RerankRequest) -> dict[str, Any]:
        if not request.documents:
            return {"results": []}
        try:
            scores = _score(gateway, api_key, request.model or model, request.query, request.documents)
        except Exception as exc:
            logger.error("scoring failed: %s", exc)
            raise HTTPException(status_code=502, detail=f"score failed: {exc}") from exc
        ranked = sorted(
            ({"index": index, "relevance_score": score} for index, score in enumerate(scores)),
            key=lambda item: item["relevance_score"],
            reverse=True,
        )
        top_n = request.top_n or len(ranked)
        return {"results": ranked[:top_n]}

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--gateway", default=os.getenv("GENERATION_API_BASE_URL", DEFAULT_GATEWAY))
    parser.add_argument("--model", default=os.getenv("GENERATION_MODEL_ALIAS", DEFAULT_MODEL))
    parser.add_argument("--api-key", default=os.getenv("GENERATION_API_KEY", ""))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.api_key:
        raise SystemExit("GENERATION_API_KEY (or --api-key) is required")
    import uvicorn

    logger.info("serving /reranks on :%d (gateway=%s model=%s)", args.port, args.gateway, args.model)
    uvicorn.run(build_app(args.gateway, args.api_key, args.model), host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
