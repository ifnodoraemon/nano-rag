"""Let the generation model take part in the FIRST-HOP retrieval.

Why this exists: the first hop was pure BM25. Everything the model contributed -
the read plan, the sufficiency check, the corrective loop - happened *after* the
candidate set was fixed. So when BM25 missed the document (measured recall@1 0.47
on the public Chinese benchmark), no amount of model reasoning could recover it:
the model never saw the document.

The model is already in the request path, so using it here adds no new model, no
embedding, and no vector index - it only changes *what the lexical index is asked
to search*.

Two mechanisms, both lexical underneath:

- **keyword expansion**: a long natural-language query is a poor BM25 query,
  because BM25 sums per-term evidence with no notion of "this whole phrase is the
  topic". The model rewrites the question into a few short keyword variants that a
  lexical index can actually match.
- **RRF fusion**: several ranked lists are merged by reciprocal rank
  (``1 / (k + rank)``) rather than by raw score. BM25 scores are not comparable
  across different queries, so fusing on rank is the standard, scale-free choice.

Structured output only: the variants come back through a JSON schema, and a
payload that violates it raises instead of being mined with regexes.
"""

from __future__ import annotations

import logging
from typing import Sequence

from app.core.exceptions import ModelOutputError
from app.model_client.generation import GenerationClient

logger = logging.getLogger(__name__)

# The paper-standard RRF constant. 60 is used widely enough that a tuned value
# would be overfitting a 200-query benchmark.
RRF_K = 60

# A keyword query is a handful of WORDS, in any language - but "word" is only countable
# where the language delimits words with spaces. A single character budget cannot serve
# both: 60 characters is a reasonable keyword query in Chinese and a truncated sentence in
# English, which is exactly why an English corpus failed 28 of 196 queries with "variant
# longer than 60 characters" - the guard was rejecting valid expansions.
#
# So the budget is per-script: a spaced script is measured in words (a keyword query is a
# handful of them however long the words are), an unspaced script in characters (it has no
# word boundary to count, and a run of 100 characters is a sentence whatever it would be
# called).
MAX_VARIANT_WORDS = 12
MAX_UNSPACED_VARIANT_CHARS = 60

EXPANSION_SCHEMA = {
    "type": "object",
    "properties": {
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "description": "3-5 short keyword queries, most useful first",
        }
    },
    "required": ["keywords"],
}

EXPANSION_PROMPT = """Rewrite the user's question into short keyword searches for a
lexical (BM25) index over document titles and bodies.

Rules:
- Each variant must be a compact keyword string (2-8 terms), NOT a sentence and
  NOT a question. Drop filler words like 请问/怎么/是什么/多少.
- Keep the domain terms, entity names and numbers exactly as written.
- Add the synonyms or formal names a document would actually use if they are
  unambiguous. Do not invent entities or numbers.
- Return at most 5 variants, most useful first.
- If the question is already a compact keyword string, return just that one.

Reply with exactly one JSON object and nothing else, using exactly this key:
{"keywords": ["<variant>", ...]}

Question: {query}
"""


async def expand_query(
    query: str,
    generation_client: GenerationClient,
    *,
    max_variants: int = 3,
) -> list[str]:
    """Ask the model for keyword variants of ``query``.

    Returns the variants only - the caller keeps the original query in the fused
    list, so expansion can add recall but can never drop what the user typed.
    """
    if max_variants <= 0:
        return []
    prompt = EXPANSION_PROMPT.replace("{query}", query)
    result = await generation_client.generate(
        [{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "query_expansion", "schema": EXPANSION_SCHEMA},
        },
    )
    content = result.get("content")
    payload = _parse_payload(content, query)
    raw = _keywords_of(payload)
    if raw is None:
        raise ModelOutputError(
            f"query expansion returned no keyword array for query {query!r}: "
            f"{content!r:.200}"
        )
    variants: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise ModelOutputError(
                f"query expansion returned a non-string variant for query {query!r}: "
                f"{item!r}"
            )
        text = " ".join(item.split()).strip()
        if not text or text.casefold() == query.casefold():
            continue
        # Measure in the unit the script actually has: words where spaces delimit them,
        # characters where they do not. Counting words alone would let a 100-character
        # Chinese sentence through as "one word"; counting characters alone rejects valid
        # English keyword queries.
        too_long, budget = _variant_exceeds_budget(text)
        if too_long:
            raise ModelOutputError(
                f"query expansion returned a {len(text)}-character variant that exceeds "
                f"its budget of {budget} for query {query!r}: {text!r}"
            )
        if text.casefold() not in {variant.casefold() for variant in variants}:
            variants.append(text)
        if len(variants) >= max_variants:
            break
    return variants


# Providers that accept response_format but do not enforce it are real (the
# project has been tested against one). They return the right content under a
# plausible key name. These aliases are accepted and logged; anything else is a
# contract violation and raises.
KEYWORD_KEYS = ("keywords", "variants", "queries", "keyword_queries")


def _variant_exceeds_budget(text: str) -> tuple[bool, str]:
    """Whether a variant is too long, judged in the unit its script provides.

    Where spaces delimit words, count words: "bank dispute substitute check refund" is a
    normal keyword query at 43 characters and would be rejected by a 60-character cap in
    a longer phrasing, while the same cap is generous in Chinese. Where the script does not
    delimit words, count characters - a run of 100 characters is a sentence whatever you
    would call its "words".
    """
    words = text.split()
    spaced = len(words) > 1 or not any(ord(ch) > 127 for ch in text)
    if spaced:
        return len(words) > MAX_VARIANT_WORDS, f"{MAX_VARIANT_WORDS} words"
    return (
        len(text) > MAX_UNSPACED_VARIANT_CHARS,
        f"{MAX_UNSPACED_VARIANT_CHARS} characters (unspaced script)",
    )


def _keywords_of(payload: dict) -> list | None:
    for key in KEYWORD_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            if key != KEYWORD_KEYS[0]:
                logger.warning(
                    "query expansion returned %r instead of 'keywords'; accepting it",
                    key,
                )
            return value
    return None


def _parse_payload(content: object, query: str) -> dict:
    from app.utils.json_utils import parse_json_object

    try:
        payload = parse_json_object(str(content or ""))
    except Exception as exc:  # noqa: BLE001 - re-raised with the query context
        raise ModelOutputError(
            f"query expansion returned unparseable output for query {query!r}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ModelOutputError(
            f"query expansion returned a non-object payload for query {query!r}"
        )
    return payload


def fuse_rankings(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """Merge ranked doc-id lists by reciprocal rank fusion.

    RRF ignores the underlying scores on purpose: a BM25 score for a keyword query
    and a BM25 score for a long question live on different scales, so adding them
    would let one variant dominate purely through its score magnitude.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + position + 1)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
