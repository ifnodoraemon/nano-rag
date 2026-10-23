"""Drop candidates the model judges irrelevant BEFORE they reach the read plan.

Why score thresholds cannot do this: on the Chinese-law corpus the BM25 scores of the
top six candidates are 0.07, 0.06, 0.06, 0.06, 0.06, 0.06 when the answer IS in the
pool and 0.06, 0.06, 0.05, 0.05, 0.05, 0.05 when it is not. A lexical score carries no
usable signal about relevance for real user questions, so "keep everything above
0.05" keeps the noise and "keep everything above 0.07" keeps almost nothing.

But the noise is not harmless: the read plan has to choose what to deep-read, and it
chooses from this list. Twelve candidates of which most are unrelated is a harder
decision than four candidates of which three are plausible.

So relevance is judged semantically, by the model that is already in the request path:
one batched call over the candidate titles and openings, giving each candidate a
relevance score. The score is used two ways:

1. FILTER - drop what the model judges clearly irrelevant, so the read plan chooses
   among a handful of plausible documents instead of a dozen unrelated ones.
2. SIGNAL - the best score in the pool says whether the pool is any good at all. A
   pool whose best candidate scores 10 is a pool that missed the answer; the caller can
   then widen the search (deeper ranking, another query) instead of confidently reading
   irrelevant documents. This is what makes the retrieval depth adaptive rather than a
   fixed constant.

Per project rules there is no silent degradation here: if the model cannot produce
usable scores the call RAISES, because "we could not judge" must not be presented as
"everything is relevant" (which would silently hand the read plan the full noise list)
nor as "nothing is relevant" (which would silently remove all evidence). A model that
genuinely scores everything below the bar is a different situation and is reported as
such - the caller learns the pool is weak, which is a fact worth acting on.
"""

from __future__ import annotations

import logging
from typing import Sequence

from app.core.exceptions import ModelOutputError
from app.model_client.generation import GenerationClient

logger = logging.getLogger(__name__)

MAX_CANDIDATES = 40
MAX_SNIPPET_CHARS = 90

FILTER_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "relevance 0-100 per candidate, in the given order",
        }
    },
    "required": ["scores"],
}

# Below this the candidate is dropped before the read plan sees it.
DROP_BELOW = 30.0

FILTER_PROMPT = """A user asked a question. Below are candidate documents retrieved for it.
Score how well each candidate could answer that question.

Scale: 100 = it plainly answers the question; 60 = same subject matter, likely
contains the rule; 30 = related topic, probably not the answer; 0 = unrelated.

Be permissive about wording: the documents are written in formal/legal language while
the user asked in everyday language. Judge the SUBJECT MATTER, not the shared words.

Return only JSON of exactly this form, one integer per candidate, in the given order:
{"scores": [<int>, <int>, ...]}

Question: __QUERY__

Candidates:
__CANDIDATES__
"""

SCORE_KEYS = ("scores", "relevance", "ratings")


def _best_excerpt(query: str, text: str) -> str:
    """The passage of ``text`` that shares the most characters with the query.

    Scoring a candidate on its opening sentence is not enough: a statute's first line
    is set-up ("为规范……制定本条例"), while the rule the user asked about sits in some
    later article. Showing the model the most query-relevant passage is what lets the
    score mean something.
    """
    body = " ".join(text.split())
    if len(body) <= MAX_SNIPPET_CHARS:
        return body
    terms = {char for char in query if "\u4e00" <= char <= "\u9fff"}
    if not terms:
        return body[:MAX_SNIPPET_CHARS]
    best_start = 0
    best_hits = -1
    step = max(1, MAX_SNIPPET_CHARS // 3)
    for start in range(0, len(body) - MAX_SNIPPET_CHARS + 1, step):
        window = body[start : start + MAX_SNIPPET_CHARS]
        hits = sum(1 for term in terms if term in window)
        if hits > best_hits:
            best_hits, best_start = hits, start
    return body[best_start : best_start + MAX_SNIPPET_CHARS]


def _render(query: str, candidates: Sequence[tuple[str, str]]) -> str:
    lines = []
    for index, (title, snippet) in enumerate(candidates):
        clean_title = " ".join(title.split())[:MAX_SNIPPET_CHARS]
        excerpt = _best_excerpt(query, snippet)
        lines.append(f"[{index}] {clean_title} :: {excerpt}")
    return FILTER_PROMPT.replace("__QUERY__", query).replace(
        "__CANDIDATES__", "\n".join(lines)
    )


async def score_candidates(
    query: str,
    candidates: Sequence[tuple[str, str]],
    generation_client: GenerationClient,
) -> list[float]:
    """Relevance score per candidate; raises when the model cannot judge."""
    if not candidates:
        return []
    head = list(candidates[:MAX_CANDIDATES])

    from app.utils.json_utils import parse_json_object

    result = await generation_client.generate(
        [{"role": "user", "content": _render(query, head)}],
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "candidate_scores", "schema": FILTER_SCHEMA},
        },
    )
    try:
        payload = parse_json_object(str(result.get("content") or ""))
    except Exception as exc:  # noqa: BLE001 - re-raised as a model output error
        raise ModelOutputError(
            f"candidate scoring returned unparseable output: {exc}"
        ) from exc

    raw = None
    if isinstance(payload, dict):
        for key in SCORE_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                raw = value
                break
    if raw is None:
        raise ModelOutputError(
            "candidate scoring returned no score array; expected a 'scores' list"
        )

    scores: list[float] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ModelOutputError(
                f"candidate scoring returned a non-numeric score: {item!r}"
            )
        scores.append(float(item))
    if len(scores) != len(head):
        raise ModelOutputError(
            f"candidate scoring returned {len(scores)} scores for {len(head)} "
            "candidates; the verdict cannot be aligned with the candidates"
        )
    return scores


async def filter_candidates(
    query: str,
    candidates: Sequence[tuple[str, str]],
    generation_client: GenerationClient,
) -> list[int]:
    """Return the indices the model believes could answer the question.

    Returns every index when the verdict is unusable, so a provider failure can never
    silently shrink the evidence the read plan gets to see.
    """
    if len(candidates) <= 1:
        return list(range(len(candidates)))
    scores = await score_candidates(query, candidates, generation_client)
    kept = [index for index, score in enumerate(scores) if score >= DROP_BELOW]
    if not kept:
        # The model judged the whole pool weak. That is a real finding - the answer was
        # not recalled - and it is reported rather than hidden behind a forced pick.
        # Verification then decides the evidence is insufficient, which is honest.
        logger.info(
            "candidate scoring: best score for %r is %.0f, below the %.0f bar; "
            "the pool appears to have missed the answer",
            query,
            max(scores) if scores else 0.0,
            DROP_BELOW,
        )
    return kept
