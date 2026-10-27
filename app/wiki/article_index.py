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

from app.core.exceptions import ModelGatewayError, ModelOutputError
from app.model_client.generation import GenerationClient

logger = logging.getLogger(__name__)

# An article shorter than this is a heading fragment, not a searchable unit. Kept low on
# purpose: "第二条 本市行政区域内国防动员的准备、实施以及相关活动，适用本条例。" is 34
# characters and is one of the most useful articles in the document (it defines scope).
# A threshold of 40 silently dropped it, and a dropped article is invisible - it simply
# never appears in retrieval results.
MIN_ARTICLE_CHARS = 15
# A trailing paragraph longer than this is content, not a heading.
HEADING_LIKE_CHARS = 40
# Sentence terminators in the scripts this system sees; a heading carries none.
SENTENCE_MARKS = ("。", "；", "！", "？", ".", ";", "!", "?")
# Above this the document is split into several requests rather than one huge prompt.
# Window size for one split request. This bounds the OUTPUT as much as the input: the
# model returns one label per article, so a 24,000-character window could ask for several
# hundred labels and the reply was truncated at the token cap - which failed the ingest
# outright on three of four real Chinese official documents. The window is sized so the
# expected reply is a few dozen labels.
MAX_DOCUMENT_CHARS = 8_000
# Floor for the retry-on-truncation loop. Below this the window holds one article and
# shrinking further would only split it.
MIN_WINDOW_CHARS = 1_500
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
- Also give `opens_with`: the first 20 characters of that unit as they appear in the
  document, so the boundary can be located exactly even when a label repeats.
- Do NOT quote the rest of the unit's body text.

Return only JSON:
{"articles": [{"label": "<label>", "opens_with": "<verbatim opening>"}, ...]}

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


def _split_embedded_heading(label: str, body: str) -> list[Article]:
    """Split a unit that ends with a heading-like line but no content after it.

    The model sometimes closes a unit at the next heading instead of starting a new unit
    there, so a section title ends up glued to the previous unit's tail - a real case in
    the manual fixture: "第二章 定期保养" sat at the end of 1.2 with nothing following.

    Recognised without vocabulary: the last paragraph of the body is SHORT, contains no
    sentence terminator, and is followed by nothing. Such a trailing line carries no
    answerable content on its own, which is exactly why leaving it attached loses a
    heading from the index while adding nothing to the unit before it.
    """
    paragraphs = [p for p in body.split("\n\n") if p.strip()]
    if len(paragraphs) < 2:
        return [Article(label=label, text=body)]
    tail = paragraphs[-1].strip()
    if len(tail) > HEADING_LIKE_CHARS or any(mark in tail for mark in SENTENCE_MARKS):
        return [Article(label=label, text=body)]
    head = "\n\n".join(paragraphs[:-1]).strip()
    if not head:
        return [Article(label=label, text=body)]
    logger.info("split trailing heading %r off unit %r", tail[:30], label[:20])
    return [Article(label=label, text=head), Article(label=tail, text=tail)]


def _locate_unit(text: str, label: str, opening: str, start: int) -> int | None:
    """Offset where this unit begins, preferring the verbatim opening.

    The opening excerpt is tried first because it is unique; the label is the fallback
    so a model that omits ``opens_with`` still works. Artificially requiring the opening
    would reject valid splits from providers that only return labels.
    """
    if opening:
        located = _find_label(text, opening, start)
        if located is not None:
            return located
    return _find_label(text, label, start)


def _find_label(text: str, label: str, start: int) -> int | None:
    r"""Offset of ``label`` at or after ``start``, tolerating layout characters.

    Two kinds of difference are layout rather than content, and both made real documents
    fail to split:

    - whitespace: the source mixes NBSP and spaces inside a number ("第十七条\xa0 国防"),
      and a form writes "1. 应用范围*" where the model transcribes "1.应用范围*（可多选）";
    - punctuation that marks a form field rather than carrying meaning: "\*", ":", ".",
      the full-width brackets a checkbox list uses.

    The pattern therefore loosens runs of whitespace AND of those layout characters,
    while still requiring every content character in the same order. A label that differs
    in its actual words still fails, which is what should happen.
    """
    import re

    # Split the label into CONTENT runs, separated by the layout characters above. Each
    # run must appear in order; the separators between them are loosened. Splitting on
    # whitespace alone missed labels that carry no spaces at all ("1.应用范围*（可多选）"),
    # which then had to match character for character including the asterisk.
    layout = set("\u00a0\t \n\r*.,:：;；、-—_()（）[]【】")
    parts: list[str] = []
    current: list[str] = []
    for ch in label:
        if ch in layout:
            if current:
                parts.append("".join(current))
                current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    if not parts:
        return None
    gap = r"[\s\u00a0\*\.,:：;；、\-—_()（）\[\]【】]*"
    pattern = re.compile(gap.join(re.escape(part) for part in parts))
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

    A reply that hits the output token cap is the one recoverable failure, and it is
    retried with a smaller window - that cap is a property of how much the model had to
    write, not of anything wrong with the document. Every other error propagates.
    """
    text = document_text.strip()
    if len(text) < MIN_ARTICLE_CHARS:
        return []
    articles: list[Article] = []
    offset = 0
    while offset < len(text):
        # Shrink the window and retry when the model's reply hits its token cap. A cap
        # is a KNOWN, recoverable condition - the reply is simply longer than the budget
        # - and a smaller window asks for fewer labels. Left alone it failed the whole
        # ingest: three of four real official documents aborted, because the reasoning
        # model's deliberation pushed the reply past the cap. Any other error still
        # raises, so a genuine gateway fault cannot be mistaken for a long document.
        window_chars = MAX_DOCUMENT_CHARS
        while True:
            window = text[offset : offset + window_chars]
            try:
                articles.extend(
                    await _split_once(window, generation_client, title=title)
                )
                break
            except ModelGatewayError as exc:
                if "finish_reason=length" not in str(exc) or window_chars <= MIN_WINDOW_CHARS:
                    raise
                window_chars = max(MIN_WINDOW_CHARS, window_chars // 2)
                logger.warning(
                    "article split for %r hit the output cap at %d chars; retrying "
                    "with %d",
                    title,
                    len(window),
                    window_chars,
                )
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
        max_tokens=24_000,
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
    openings: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            # Tolerate the older shape a provider may still return.
            label = item.get("label") or item.get("starts_with")
            opening = item.get("opens_with")
        else:
            label, opening = item, None
        if not isinstance(label, str) or not label.strip():
            raise ModelOutputError(
                f"article split returned an unusable entry for {title!r}: {item!r}"
            )
        anchors.append((label.strip(), label.strip()))
        openings.append(opening.strip() if isinstance(opening, str) else "")

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
    for index, (_anchor, label) in enumerate(anchors):
        if not label:
            raise ModelOutputError(
                f"article split returned an entry without a label for {title!r}; "
                "the label is what locates the article in the document"
            )
        # Prefer the verbatim opening when the model supplied one: a label alone can
        # repeat (two units both starting "第二章", or a contents entry), and resolving
        # the wrong repeat silently shifts every following boundary.
        opening = openings[index] if index < len(openings) else ""
        located = _locate_unit(text, label, opening, cursor)
        if located is None:
            raise ModelOutputError(
                f"article {label!r} was not found in {title!r} after offset "
                f"{cursor}; the model named a unit the document does not contain"
            )
        positions.append((located, label))
        cursor = located + 1

    articles: list[Article] = []
    for index, (position, label) in enumerate(positions):
        end = positions[index + 1][0] if index + 1 < len(positions) else len(text)
        raw_body = text[position:end]
        body = raw_body.strip()
        if len(body) < MIN_ARTICLE_CHARS:
            # A fragment this short is a heading or a stray line; indexing it would add a
            # document that can never be the right answer on its own.
            continue
        articles.extend(_split_embedded_heading(label, body))
    if not articles:
        raise ModelOutputError(
            f"article split produced no article above {MIN_ARTICLE_CHARS} characters "
            f"for {title!r}"
        )
    logger.info("split %r into %d article(s)", title, len(articles))
    return articles
