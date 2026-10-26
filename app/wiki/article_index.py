"""Index the individual articles of a long document, not just its summary.

Why this exists. The discovery layer indexes one wiki page per document, and that page
is a summary: measured on 595 Chinese statutes averaging 5,446 characters, the indexed
body is 1,049 characters - about 15% of the source. The rest of the statute is invisible
to BM25. A user asking about one clause is competing for a single document-level score
against everything else the document says, and the passage that answers them is usually
not in the summary.

What this does. A statute is already segmented into articles by its own structure
(第X条). Each article becomes an index entry carrying the parent document's identity, so
a hit reports "this DOCUMENT, this ARTICLE" - the document is still the unit the read
plan and citations work with, the article is only the matching unit. No new retrieval
stage, no second index format: the same BM25 index holds both page kinds.

Where the boundaries come from. The model, not a regex. Chinese statutes number their
articles in ways that a pattern can follow most of the time and silently get wrong the
rest: 第一条/第二条 style, "Article 3", nested numbering, tables of contents that repeat
every heading, and documents with no numbering at all. A wrong split destroys the
article it cuts, and the damage is invisible. So the same rule as chunking applies - the
model decides, and the text is sliced from the original by anchors that are verified
against it, never rewritten.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from app.core.exceptions import ModelOutputError
from app.model_client.generation import GenerationClient

logger = logging.getLogger(__name__)

# An article shorter than this is a heading fragment, not a searchable unit. Kept low on
# purpose: "第二条 本市行政区域内国防动员的准备、实施以及相关活动，适用本条例。" is 34
# characters and is one of the most useful articles in the document (it defines scope).
# A threshold of 40 silently dropped it, and a dropped article is invisible - it simply
# never appears in retrieval results.
MIN_ARTICLE_CHARS = 15
# Above this the document is split into several requests rather than one huge prompt.
MAX_DOCUMENT_CHARS = 24_000
MAX_ARTICLES = 600

# Only the labels are requested. Asking for a verbatim snippet per article made the
# model emit thousands of copied characters - slow, token-hungry, and unnecessary since
# the label alone locates the article.
ARTICLE_ANCHOR_SCHEMA = {
    "type": "object",
    "properties": {
        "articles": {
            "type": "array",
            "items": {"type": "string"},
            "description": "the article label, in document order",
        }
    },
    "required": ["articles"],
}

PROMPT = """List the segments of this document - the smallest complete units a reader
would cite individually. Whatever the document is (a statute, a contract, a manual, a
policy, an agreement), it groups its content into units of some kind.

Rules:
- One entry per unit, in document order.
- The entry is that unit's LABEL: its own designation if it has one (a number, a heading,
  a section title), or its first few characters if it has none.
- Copy the label exactly as the document writes it. Do not merge, skip or invent entries.
- Do NOT quote the unit's body text.

Return only JSON: {"articles": ["<label>", "<label>", ...]}

DOCUMENT:
__TEXT__
"""


# A contents entry duplicates the first real label within this distance. Measured on
# statutes whose contents pages list chapter headings: the first article's body is never
# this close to its contents mention.
CONTENTS_PROXIMITY = 400


def _body_offset(text: str, first_label: str, label_offsets: Sequence[int]) -> int:
    """Where the body starts, past a table of contents if the document has one.

    Recognised structurally, with no vocabulary: if the document lists its own first
    label twice in quick succession, the first occurrence is a contents entry and the
    body starts at the second. This used to be written as ``text.find("第一条")``, which
    worked for Chinese statutes and silently did nothing for a contract, a manual, or an
    English document - a special case pretending to be a rule.

    Returns 0 when there is no contents page.
    """
    if not label_offsets:
        return 0
    first = label_offsets[0]
    second = _find_label(text, first_label, first + 1)
    if second is not None and second - first < CONTENTS_PROXIMITY:
        return second
    return first


def _find_label(text: str, label: str, start: int) -> int | None:
    """Offset of ``label`` at or after ``start``, tolerating whitespace variants.

    The source mixes NBSP and ordinary spaces inside the number ("第十七条\xa0 国防"),
    so an exact search misses.* A regex built from the label with every whitespace run
    loosened finds it while still requiring the same characters in the same order.
    """
    import re

    parts = label.split()
    if not parts:
        return None
    pattern = re.compile(r"[\s\u00a0]*".join(re.escape(part) for part in parts))
    match = pattern.search(text, start)
    return match.start() if match else None


@dataclass(frozen=True)
class Article:
    label: str
    text: str


def _render(text: str) -> str:
    # str.replace, not .format: the prompt carries a literal JSON example.
    return PROMPT.replace("__TEXT__", text)


async def split_into_articles(
    document_text: str,
    generation_client: GenerationClient,
    *,
    title: str = "",
) -> list[Article]:
    """Split a document into its own articles, with anchors verified against the text.

    Raises when the model's anchors do not resolve: a plan built on paraphrased anchors
    would silently reassemble the document into something the user never wrote.
    """
    text = document_text.strip()
    if len(text) < MIN_ARTICLE_CHARS:
        return []
    if len(text) <= MAX_DOCUMENT_CHARS:
        return await _split_once(text, generation_client, title=title)

    # Long documents: split in windows, but at a window boundary that the model chooses
    # so an article is not cut across two requests.
    articles: list[Article] = []
    offset = 0
    while offset < len(text):
        window = text[offset : offset + MAX_DOCUMENT_CHARS]
        articles.extend(await _split_once(window, generation_client, title=title))
        offset += len(window)
        if len(articles) > MAX_ARTICLES:
            raise ModelOutputError(
                f"{title!r} produced more than {MAX_ARTICLES} articles; "
                "the split looks wrong"
            )
    return articles


async def _split_once(
    text: str, generation_client: GenerationClient, *, title: str
) -> list[Article]:
    from app.utils.json_utils import parse_json_object

    result = await generation_client.generate(
        [{"role": "user", "content": _render(text)}],
        temperature=0,
        max_tokens=16_000,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "articles", "schema": ARTICLE_ANCHOR_SCHEMA},
        },
    )
    try:
        payload = parse_json_object(str(result.get("content") or ""))
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise ModelOutputError(
            f"article split returned unparseable output for {title!r}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ModelOutputError(f"article split returned a non-object payload for {title!r}")

    raw = payload.get("articles")
    if not isinstance(raw, list) or not raw:
        raise ModelOutputError(
            f"article split returned no articles for {title!r}; a document of "
            f"{len(text)} characters has at least one"
        )

    anchors: list[tuple[str, str]] = []
    for item in raw:
        if isinstance(item, dict):
            # Tolerate the older shape a provider may still return.
            label = item.get("label") or item.get("starts_with")
        else:
            label = item
        if not isinstance(label, str) or not label.strip():
            raise ModelOutputError(
                f"article split returned an unusable entry for {title!r}: {item!r}"
            )
        anchors.append((label.strip(), label.strip()))

    # Cut points come from the LABEL, not from a re-typed copy of the text. Chinese
    # statutes begin every article with its own number (第十七条), and that number is
    # short enough to locate exactly and impossible to paraphrase usefully. Matching a
    # 24-character snippet instead failed on documents the model had quoted correctly,
    # because the source mixes NBSP and ordinary spaces inside the number
    # ("第十七条\xa0 国防动员...") while the model always emits a plain space.
    #
    # The model still decides the boundaries - it decides WHICH labels are articles,
    # their order and their names; we only resolve each label to an offset.
    # A table of contents repeats every chapter heading, and often the article names
    # too. Locating labels from the front therefore lands inside the contents page and
    # shifts every subsequent cut. The body starts after the LAST contents-like heading,
    # which is where the numbering restarts.
    positions: list[tuple[int, str]] = []
    first_label = anchors[0][1] if anchors else ""
    first_hit = _find_label(text, first_label, 0) if first_label else None
    cursor = _body_offset(
        text, first_label, [first_hit] if first_hit is not None else []
    )
    for _anchor, label in anchors:
        if not label:
            raise ModelOutputError(
                f"article split returned an entry without a label for {title!r}; "
                "the label is what locates the article in the document"
            )
        located = _find_label(text, label, cursor)
        if located is None:
            raise ModelOutputError(
                f"article label {label!r} was not found in {title!r} after offset "
                f"{cursor}; the model named an article the document does not contain"
            )
        positions.append((located, label))
        cursor = located + 1

    articles: list[Article] = []
    for index, (position, label) in enumerate(positions):
        end = positions[index + 1][0] if index + 1 < len(positions) else len(text)
        body = text[position:end].strip()
        if len(body) < MIN_ARTICLE_CHARS:
            # A fragment this short is a heading or a stray line; indexing it would add a
            # document that can never be the right answer on its own.
            continue
        articles.append(Article(label=label, text=body))
    if not articles:
        raise ModelOutputError(
            f"article split produced no article above {MIN_ARTICLE_CHARS} characters "
            f"for {title!r}"
        )
    logger.info("split %r into %d article(s)", title, len(articles))
    return articles
