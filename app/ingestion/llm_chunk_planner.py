"""Let the model decide where a document should be cut.

Why the model and not a rule: cutting a regulation or contract into chunks is a
judgement about *meaning* - "does this piece stand on its own as an answerable
unit?" - and every rule for it is a hardcoded guess about typography. Matching
``第X条`` works until a document numbers clauses differently, or comes out of a PDF as
one undifferentiated paragraph, or is a manual with no numbering at all. That is
exactly the class of hardcoded semantic rule this project forbids elsewhere.

The model therefore does not rewrite the document. It returns *boundaries*: for each
new chunk, the opening words where that chunk should start. The text itself is sliced
out of the original by those markers, so:

- the model cannot drop, paraphrase or hallucinate content;
- every boundary is verified against the original text before it is used;
- if the model returns a marker that does not occur, or the markers are out of order,
  the plan is rejected loudly instead of producing mangled chunks.

This keeps the splitter honest: the model supplies judgement, the code enforces that
nothing was lost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from app.core.exceptions import ModelOutputError
from app.model_client.generation import GenerationClient

logger = logging.getLogger(__name__)

# How many words of the target chunk start are sent/returned as the anchor. Long
# enough to be unique inside one document, short enough to survive model tokenisation.
ANCHOR_CHARS = 24
MAX_CHUNKS_PER_DOCUMENT = 400

BOUNDARY_SCHEMA = {
    "type": "object",
    "properties": {
        "chunks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "starts_with": {"type": "string"},
                    "heading": {"type": "string"},
                },
                "required": ["starts_with"],
            },
        }
    },
    "required": ["chunks"],
}

BOUNDARY_PROMPT = """You are splitting a document into retrieval chunks.

Split ONLY at boundaries where the text stays semantically self-contained: a whole
clause, article, section, or self-contained paragraph. Never cut a clause, sentence,
table or numbered item in half. Keep each chunk under __MAX_CHARS__ characters where
the document's own structure allows it.

Do not rewrite anything. For every chunk, output the FIRST __ANCHOR__ CHARACTERS of
that chunk exactly as they appear in the document (verbatim, no ellipsis), so the text
can be located by string match. The first chunk must start at the very beginning of the
document.

Return only JSON of exactly this form:
{"chunks": [{"starts_with": "<verbatim opening characters>", "heading": "<its heading, or empty>"}, ...]}

DOCUMENT:
__DOCUMENT__
"""

ANCHOR_KEYS = ("starts_with", "start", "anchor")


@dataclass(frozen=True)
class ChunkBoundary:
    start: int
    heading: str


def _render_prompt(text: str, max_chars: int) -> str:
    # str.replace, not .format: the template contains a literal JSON example.
    return (
        BOUNDARY_PROMPT.replace("__MAX_CHARS__", str(max_chars))
        .replace("__ANCHOR__", str(ANCHOR_CHARS))
        .replace("__DOCUMENT__", text)
    )


def _anchors_of(payload: dict, text: str) -> list[tuple[str, str]]:
    for key in ("chunks", "boundaries", "segments"):
        raw = payload.get(key)
        if isinstance(raw, list):
            if key != "chunks":
                logger.warning("chunk planning returned %r instead of 'chunks'", key)
            break
    else:
        raise ModelOutputError(
            "chunk planning returned no chunk list; expected a 'chunks' array"
        )
    anchors: list[tuple[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ModelOutputError(
                f"chunk planning returned a non-object entry: {item!r}"
            )
        anchor = ""
        for key in ANCHOR_KEYS:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                anchor = value.strip()
                break
        if not anchor:
            raise ModelOutputError(
                f"chunk planning entry has no verbatim anchor: {item!r}"
            )
        heading = item.get("heading")
        anchors.append((anchor, str(heading).strip() if heading else ""))
    return anchors


def locate_boundaries(text: str, anchors: Sequence[tuple[str, str]]) -> list[ChunkBoundary]:
    """Resolve model-supplied anchors to offsets, verifying each one.

    Verification is the point: a marker that cannot be found exactly once, or that
    moves backwards, means the model paraphrased or invented content, and a chunk plan
    built on that would silently reassemble the document into something the user never
    wrote.
    """
    boundaries: list[ChunkBoundary] = []
    cursor = 0
    for anchor, heading in anchors:
        index = text.find(anchor, cursor)
        if index < 0:
            raise ModelOutputError(
                f"chunk planning anchor not found after offset {cursor}: {anchor!r}. "
                "The model paraphrased the document instead of quoting it."
            )
        if index < cursor:
            raise ModelOutputError(
                f"chunk planning anchor out of order at offset {cursor}: {anchor!r}"
            )
        boundaries.append(ChunkBoundary(start=index, heading=heading))
        cursor = index + 1
    if not boundaries:
        raise ModelOutputError("chunk planning produced no boundaries")
    if boundaries[0].start != 0:
        # The document's opening would otherwise belong to no chunk.
        boundaries.insert(0, ChunkBoundary(start=0, heading=""))
    return boundaries


def slice_by_boundaries(text: str, boundaries: Sequence[ChunkBoundary]) -> list[tuple[str, str]]:
    """Cut ``text`` at the resolved offsets. Concatenation equals the original."""
    pieces: list[tuple[str, str]] = []
    for index, boundary in enumerate(boundaries):
        end = boundaries[index + 1].start if index + 1 < len(boundaries) else len(text)
        piece = text[boundary.start : end].strip()
        if piece:
            pieces.append((piece, boundary.heading))
    return pieces


async def plan_chunk_boundaries(
    text: str,
    generation_client: GenerationClient,
    *,
    max_chars: int,
) -> list[tuple[str, str]]:
    """Ask the model where to cut, then slice the original text there.

    Returns ``[(chunk_text, heading)]``. The chunk texts are contiguous slices of the
    input, so no content is added, dropped or reordered.
    """
    value = text.strip()
    if not value:
        return []
    if len(value) <= max_chars:
        return [(value, "")]

    from app.utils.json_utils import parse_json_object

    result = await generation_client.generate(
        [{"role": "user", "content": _render_prompt(value, max_chars)}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "chunk_plan", "schema": BOUNDARY_SCHEMA},
        },
    )
    content = result.get("content")
    try:
        payload = parse_json_object(str(content or ""))
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise ModelOutputError(
            f"chunk planning returned unparseable output: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ModelOutputError("chunk planning returned a non-object payload")
    anchors = _anchors_of(payload, value)
    if len(anchors) > MAX_CHUNKS_PER_DOCUMENT:
        raise ModelOutputError(
            f"chunk planning asked for {len(anchors)} chunks, above the "
            f"{MAX_CHUNKS_PER_DOCUMENT} limit"
        )
    boundaries = locate_boundaries(value, anchors)
    return slice_by_boundaries(value, boundaries)
