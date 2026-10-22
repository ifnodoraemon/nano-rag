"""Reorder the first-hop candidates with the same generation model.

Why this exists: query expansion improved *deep* recall (recall@12, +3.1pt on two
independent samples) but not top-1 ranking, and recall@1 stayed near 0.49. The
remaining lever is ordering, and the project's constraint is that no separate
reranker may be introduced - so the rerank is done by the generation model that is
already in the request path.

The shape of the call matters for cost. Ranking every candidate against every other
pair is quadratic; ranking them in one listwise pass is a single call, and the
existing lexical order supplies a prior the model can lean on (the same insight as
"Setwise Insertion", arXiv:2504.10509, which reuses the initial ranking instead of
re-deriving it). Its measured 23% inference reduction over per-comparison prompting
is why a one-call listwise rerank is the right default here.

The contract is strict: the reply must be a permutation of the candidate indices.
A partial, duplicated or out-of-range ordering raises instead of being patched up -
silently "fixing" it would reorder documents by a rule the model never expressed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from app.core.exceptions import ModelOutputError
from app.model_client.generation import GenerationClient

logger = logging.getLogger(__name__)

MAX_SNIPPET_CHARS = 120

RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "order": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "indices of the candidates, most relevant first",
        }
    },
    "required": ["order"],
}

RERANK_PROMPT = """Rank the candidate documents by how well each answers the question.
Use the document title and the opening text; judge usefulness to the person who
asked, not surface word overlap.

Return ONLY a JSON object of exactly this form, using every index exactly once:
{"order": [<index>, <index>, ...]}   (most relevant first)

Question: __QUERY__

Candidates:
__CANDIDATES__
"""

# Providers that accept response_format but do not enforce it may rename the key.
ORDER_KEYS = ("order", "ranking", "ranked_indices", "indices")


@dataclass(frozen=True)
class RerankDocument:
    doc_id: str
    title: str
    text: str


def _render_candidates(documents: Sequence[RerankDocument]) -> str:
    lines = []
    for index, document in enumerate(documents):
        title = " ".join(document.title.split())[:MAX_SNIPPET_CHARS]
        snippet = " ".join(document.text.split())[:MAX_SNIPPET_CHARS]
        lines.append(f"[{index}] {title} :: {snippet}")
    return "\n".join(lines)


def _render_prompt(query: str, documents: Sequence[RerankDocument]) -> str:
    # str.replace, not .format: the template contains a literal JSON example whose
    # braces would be read as format fields.
    return RERANK_PROMPT.replace("__QUERY__", query).replace(
        "__CANDIDATES__", _render_candidates(documents)
    )


def _order_of(payload: dict) -> list | None:
    for key in ORDER_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            if key != ORDER_KEYS[0]:
                logger.warning("rerank returned %r instead of 'order'; accepting it", key)
            return value
    return None


def _validate_permutation(raw: list, expected: int, query: str) -> list[int]:
    """The reply must visit every index exactly once."""
    order: list[int] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ModelOutputError(
                f"rerank returned a non-integer index for query {query!r}: {item!r}"
            )
        if item < 0 or item >= expected:
            raise ModelOutputError(
                f"rerank index {item} is out of range 0..{expected - 1} for query {query!r}"
            )
        if item in order:
            raise ModelOutputError(
                f"rerank repeated index {item} for query {query!r}; the ordering must "
                "be a permutation of the candidates"
            )
        order.append(item)
    if len(order) != expected:
        raise ModelOutputError(
            f"rerank returned {len(order)} of {expected} candidate indices for query "
            f"{query!r}; a partial ordering would silently drop candidates"
        )
    return order


async def rerank_documents(
    query: str,
    documents: Sequence[RerankDocument],
    generation_client: GenerationClient,
    *,
    top_n: int,
) -> list[str]:
    """Return doc ids ordered by the model, most relevant first.

    Only the first ``top_n`` candidates are sent: the rerank is meant to fix the
    head of the ranking, and sending the whole candidate list costs tokens for
    documents the read plan would never reach. The untouched tail keeps its lexical
    order and is appended, so no candidate is ever lost.
    """
    if len(documents) <= 1 or top_n <= 0:
        return [document.doc_id for document in documents]
    head = list(documents[:top_n])
    tail = list(documents[top_n:])

    from app.utils.json_utils import parse_json_object

    result = await generation_client.generate(
        [{"role": "user", "content": _render_prompt(query, head)}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "rerank", "schema": RERANK_SCHEMA},
        },
    )
    content = result.get("content")
    try:
        payload = parse_json_object(str(content or ""))
    except Exception as exc:  # noqa: BLE001 - re-raised with the query context
        raise ModelOutputError(
            f"rerank returned unparseable output for query {query!r}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ModelOutputError(f"rerank returned a non-object payload for query {query!r}")
    raw = _order_of(payload)
    if raw is None:
        raise ModelOutputError(
            f"rerank returned no ordering array for query {query!r}: {content!r:.200}"
        )
    order = _validate_permutation(raw, len(head), query)
    reordered = [head[index].doc_id for index in order]
    return reordered + [document.doc_id for document in tail]
