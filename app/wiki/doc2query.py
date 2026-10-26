"""Generate the questions a document answers, and index them with it.

The measurement that motivates this: on 2,996 real Chinese statutes, queries taken
verbatim from a clause retrieve that statute with hit@1 = 0.87, while real user
questions retrieve it with hit@1 = 0.12. Same index, same stack - only the phrasing
changes. A user says "公司没给我做安全培训就让我直接上岗，这样允许吗？" and the statute says
"生产经营单位应当对从业人员进行安全生产教育和培训". A lexical index has almost nothing to
match on.

Teaching the QUERY to sound like the document (query expansion) helps but is limited by
having to guess the wording at query time, every time. This module does the mirror
image: at ingest time, ask the model which everyday questions a document answers, and
store those questions in the document's wiki page so the lexical index can match them.
The cost is paid once per document instead of once per query, and a question generated
by a model that has read the WHOLE document is better informed than one generated from a
user's short query.

Why not just summarise: a summary keeps the document's vocabulary. Synthetic questions
use the QUESTIONER's vocabulary - the exact gap being closed.

Storage choice: the questions go into the source page body, which is what the BM25
discovery layer already indexes. No new index, no new store, no extra runtime stage.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from app.core.exceptions import ModelOutputError
from app.model_client.generation import GenerationClient

logger = logging.getLogger(__name__)

# How many questions per document. Higher means more query-side surface but a longer
# page body, which dilutes BM25 term statistics; 8 is a starting point to be measured.
QUESTIONS_PER_DOCUMENT = 8
# Text sent to the model. Longer documents are sampled by section so the questions
# cover the whole document instead of only its opening.
MAX_SOURCE_CHARS = 12_000
MAX_QUESTION_CHARS = 60

GENERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "everyday questions this document answers",
        }
    },
    "required": ["questions"],
}

PROMPT = """Read the document below (a Chinese statute, regulation, contract or manual)
and write the questions a normal person would ask that THIS document answers.

Rules:
- Use everyday spoken Chinese. Do NOT copy legal phrasing, do NOT quote clause numbers,
  do NOT mention the document's title.
- Each question must be answerable from the text below and must be specific enough that
  a different document would not answer it.
- Cover different parts of the document, not eight variations of one clause.
- One question per entry, at most __MAX_CHARS__ characters.

Return only JSON: {"questions": ["...", "..."]}
Produce between 4 and __COUNT__ questions.

DOCUMENT:
__TEXT__
"""

QUESTION_KEYS = ("questions", "queries", "faq")


def _render(document_text: str) -> str:
    # str.replace, not .format: the prompt carries a literal JSON example.
    return (
        PROMPT.replace("__MAX_CHARS__", str(MAX_QUESTION_CHARS))
        .replace("__COUNT__", str(QUESTIONS_PER_DOCUMENT))
        .replace("__TEXT__", document_text[:MAX_SOURCE_CHARS])
    )


def _sample_sections(text: str) -> str:
    """Span the whole document when it is too long to send whole.

    Taking the first N characters would generate questions only about the preamble -
    the least query-relevant part of a statute.
    """
    if len(text) <= MAX_SOURCE_CHARS:
        return text
    window = MAX_SOURCE_CHARS // 4
    starts = [
        int((len(text) - window) * fraction) for fraction in (0.0, 0.3, 0.6, 1.0)
    ]
    return "\n\n[...]\n\n".join(
        text[start : start + window].strip() for start in sorted(set(starts))
    )


def _clean_questions(raw: Sequence[Any]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            raise ModelOutputError(
                f"synthetic question generation returned a non-string entry: {item!r}"
            )
        question = " ".join(item.split()).strip()
        if not question:
            continue
        if len(question) > MAX_QUESTION_CHARS * 2:
            # A rambling paragraph is not a question; it is boilerplate the model
            # copied from the document, which would pollute the index.
            logger.warning("dropping over-long synthetic question: %r", question[:80])
            continue
        if question in seen:
            continue
        seen.add(question)
        cleaned.append(question)
    return cleaned[:QUESTIONS_PER_DOCUMENT]


async def generate_questions(
    document_text: str,
    generation_client: GenerationClient,
    *,
    title: str = "",
) -> list[str]:
    """Questions a user could ask that this document answers.

    Raises ``ModelOutputError`` when the model returns nothing usable: a silent empty
    list would look identical to "this document answers nothing", which is not a
    conclusion this stage is entitled to reach.
    """
    text = _sample_sections(document_text.strip())
    if not text:
        return []

    from app.utils.json_utils import parse_json_object

    result = await generation_client.generate(
        [{"role": "user", "content": _render(text)}],
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "synthetic_questions", "schema": GENERATION_SCHEMA},
        },
    )
    try:
        payload = parse_json_object(str(result.get("content") or ""))
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise ModelOutputError(
            f"synthetic question generation returned unparseable output for {title!r}: "
            f"{exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ModelOutputError(
            f"synthetic question generation returned a non-object payload for {title!r}"
        )
    raw = None
    for key in QUESTION_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            raw = value
            break
    if raw is None:
        raise ModelOutputError(
            f"synthetic question generation returned no question list for {title!r}"
        )
    if not raw:
        raise ModelOutputError(
            f"synthetic question generation returned an empty list for {title!r}; "
            "every document answers at least one question"
        )
    return _clean_questions(raw)
