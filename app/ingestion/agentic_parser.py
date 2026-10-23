"""Parse a document with the model driving three format-agnostic tools.

The model does not get a tool per file type. It gets ``inspect_document``,
``read_text`` and ``read_image`` - see ``document_access`` - so the same protocol works
for a DOCX notice, a scanned PDF, a spreadsheet, a slide deck or a photo of a printed
form. What differs between those is the adapter behind the tool, not the model's
vocabulary.

Why an agent loop at all: real Chinese official documents carry no Word heading styles;
the entire hierarchy lives in the numbering inside the text (一、/（一）/1.). A single
"convert this file to Markdown" call reproduced that hierarchy in some runs and flattened
it in others, and a flattened tree later made the read plan name a section that did not
exist. Letting the model inspect the parts and reason first makes the structure
something it established from evidence rather than guessed in one pass.

Images are injected as real image content in a follow-up user turn: a tool result is
text-only, and "an image is at part pg3" would ask the model to imagine the page.
"""

from __future__ import annotations

import base64
import json
import os
import logging
import re
from pathlib import Path
from typing import Any

from typing import TYPE_CHECKING

from app.core.exceptions import ModelGatewayError, ModelOutputError, ParsingError
from app.ingestion.document_access import TOOL_SCHEMAS, DocumentAccess
if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.ingestion.document_access import Document

from app.model_client.capabilities import ModelCapabilities, probe_capabilities
from app.model_client.generation import GenerationClient

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 14
MAX_IMAGES_PER_DOCUMENT = 8

SYSTEM_PROMPT = """You reconstruct the structure of business documents: Chinese official
notices (通知), regulations, contracts, manuals, application forms, spreadsheets, slides
and scanned pages.

You have three tools: inspect_document, read_text and read_image. Use them; never guess
what a document contains.

Produce the document as Markdown:
- `#` for the document title, `##` for the first real section level, `###` for the next,
  and so on, matching the document's own nesting.
- Body text verbatim. Do not paraphrase, summarise, reorder, merge or drop anything.
- Tables as Markdown tables; a spreadsheet sheet becomes a section with its table.
- Scanned pages: read them as images and transcribe what they say.

These documents frequently have no Word heading styles at all and express hierarchy only
through numbering in the text (一、/（一）/1./（1）or other conventions). Judge the depth
yourself, as a careful editor would, from the content and numbering. Different documents
use different conventions - adapt to each one.

When finished, reply with the Markdown only: no commentary, no code fence."""

FINAL_INSTRUCTION = """Output the finished document now.

Reply with only JSON: {"markdown": "<the whole document with heading markers>"}
Body text verbatim, headings at their true depth."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"markdown": {"type": "string"}},
    "required": ["markdown"],
}


async def parse_document(
    path_str: str,
    generation_client: GenerationClient,
    access: DocumentAccess,
) -> str:
    """Drive the tool loop and return reconstructed Markdown."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Reconstruct this document: {path_str}\n"
                "Start by inspecting it."
            ),
        },
    ]
    images_injected = 0
    for round_index in range(MAX_TOOL_ROUNDS):
        data = await generation_client.generate(
            messages, temperature=0, tools=TOOL_SCHEMAS, tool_choice="auto"
        )
        message = (data.get("choices") or [{}])[0].get("message") or {}
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            content = str(message.get("content") or "").strip()
            if content:
                return await _finalise(generation_client, content)
            break

        messages.append({"role": "assistant", **message})
        image_parts: list[dict[str, Any]] = []
        for call in tool_calls:
            result_message, image = _execute_tool(call, access)
            messages.append(result_message)
            if image is not None:
                image_parts.append(image)
        if image_parts:
            images_injected += len(image_parts)
            if images_injected > MAX_IMAGES_PER_DOCUMENT:
                raise ParsingError(
                    f"{path_str} needs more than {MAX_IMAGES_PER_DOCUMENT} page images; "
                    "refusing to keep pulling scans into one request"
                )
            messages.append(_image_message(image_parts))
        logger.info(
            "agentic parse round %d: %d tool call(s), %d image(s)",
            round_index + 1,
            len(tool_calls),
            len(image_parts),
        )

    messages.append({"role": "user", "content": FINAL_INSTRUCTION})
    data = await generation_client.generate(
        messages,
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "document_markdown", "schema": OUTPUT_SCHEMA},
        },
    )
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
    return _extract_markdown(content, path_str)


def _execute_tool(
    call: dict[str, Any], access: DocumentAccess
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    function = call.get("function") or {}
    name = str(function.get("name") or "")
    raw_args = function.get("arguments") or "{}"
    try:
        arguments = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
    except json.JSONDecodeError as exc:
        raise ModelOutputError(
            f"tool call {name} carried unparseable arguments: {exc}"
        ) from exc

    tool_call_id = call.get("id") or name
    image: dict[str, Any] | None = None
    try:
        if name == "inspect_document":
            result = access.inspect(**arguments)
        elif name == "read_text":
            result = access.read_text(**arguments)
        elif name == "read_image":
            result = access.read_image(**arguments)
            if "base64" in result:
                image = {
                    "mime": result.get("mime", "image/png"),
                    "base64": result.pop("base64"),
                    "part_id": result.get("part_id"),
                }
        else:
            raise ParsingError(f"unknown tool {name!r}")
    except TypeError as exc:
        result = {"error": f"bad arguments for {name}: {exc}"}
    except ParsingError as exc:
        # A refusal is information the model can act on (wrong part id, unreadable
        # page); a crash would not be. Only domain errors become tool results.
        result = {"error": str(exc)}
    return (
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(result, ensure_ascii=False)[:80_000],
        },
        image,
    )


def _image_message(images: list[dict[str, Any]]) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": "Rendered document parts requested above:",
        }
    ]
    for image in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image['mime']};base64,{image['base64']}"},
            }
        )
    return {"role": "user", "content": content}


async def _finalise(generation_client: GenerationClient, draft: str) -> str:
    markdown = _strip_fence(draft)
    if not markdown.strip():
        raise ModelOutputError("the document parser returned an empty document")
    return markdown


def _extract_markdown(content: object, path_str: str) -> str:
    from app.utils.json_utils import parse_json_object

    try:
        payload = parse_json_object(str(content or ""))
    except Exception as exc:  # noqa: BLE001 - re-raised with the document name
        raise ModelOutputError(
            f"the document parser returned unparseable output for {path_str}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ModelOutputError(
            f"the document parser returned a non-object payload for {path_str}"
        )
    markdown = payload.get("markdown")
    if not isinstance(markdown, str) or not markdown.strip():
        raise ModelOutputError(
            f"the document parser returned no 'markdown' for {path_str}"
        )
    return _strip_fence(markdown)


_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n?```\s*$", re.DOTALL)


def _strip_fence(text: str) -> str:
    match = _FENCE_RE.match(text)
    return match.group(1) if match else text


# --------------------------------------------------------------------- strategy

# Document text per request when the model cannot drive tools. Kept well below the
# context limit because the OUTPUT is roughly as large as the input (the document is
# rewritten as Markdown) and a reasoning model also spends tokens deliberating first.
INLINE_BUDGET_CHARS = 6_000
# Output budget for one structuring call. This project optimises for quality, not cost.
STRUCTURE_MAX_TOKENS = 32_000


def choose_strategy(
    document: "Document", capabilities: "ModelCapabilities"
) -> tuple[str, str]:
    """Pick a strategy from the DOCUMENT and the MODEL - never from the file format.

    No layer assumes anything: the same code serves a 3 KB notice and a 300-page manual,
    a model with native tool calling and one without, a model that reads images and one
    that does not. Returns ``(strategy, reason)`` where strategy is one of
    ``unsupported`` / ``tool_loop`` / ``inline`` / ``staged``.
    """
    text_parts = [part for part in document.parts if part.kind == "text"]
    image_parts = [part for part in document.parts if part.kind == "image"]
    total_chars = sum(part.size_hint() for part in text_parts)

    if image_parts and not capabilities.can_read_images:
        return (
            "unsupported",
            f"{len(image_parts)} image part(s) need a vision-capable model; the "
            "configured model has not been measured as able to read images",
        )
    # Tool calling is OPT-IN, not the default. A provider can pass a capability probe
    # and still degrade at the real prompt: this gateway answered a simple probe with
    # structured tool_calls, then emitted DSML tool syntax as plain text on the actual
    # document. Inline/staged does not depend on that protocol at all, so it is the
    # general baseline; the loop is enabled per deployment once it is known to hold.
    if capabilities.supports_tools is True and capabilities_allow_tool_loop():
        return (
            "tool_loop",
            "model returns structured tool calls and the tool loop is enabled",
        )
    if total_chars <= INLINE_BUDGET_CHARS:
        return (
            "inline",
            f"no native tool calling; {total_chars} chars fit in one request",
        )
    return (
        "staged",
        f"no native tool calling and {total_chars} chars exceed one request; "
        "the document is structured piece by piece",
    )


async def parse_adaptive(
    path_str: str,
    generation_client: GenerationClient,
    access: "DocumentAccess",
    capabilities: "ModelCapabilities",
) -> str:
    """Parse using whatever the document and the model together allow."""
    document = access.document(path_str)
    strategy, reason = choose_strategy(document, capabilities)
    logger.info("parse strategy for %s: %s (%s)", path_str, strategy, reason)

    if strategy == "unsupported":
        raise ParsingError(f"cannot parse {path_str}: {reason}")
    if strategy == "tool_loop":
        return await parse_document(path_str, generation_client, access)
    if strategy == "inline":
        return await _parse_inline(document, generation_client, path_str)
    return await _parse_staged(document, generation_client, path_str)


async def _parse_inline(
    document: "Document", generation_client: GenerationClient, path_str: str
) -> str:
    """One request carrying every text part, plus the images when the model sees.

    A guide that is 134 characters of text and five screenshots is a normal document,
    not an edge case: the images ARE the content, so they travel with the text in the
    same request.
    """
    blocks = [
        f"--- part {part.part_id} ({part.label}) ---\n{part.text or ''}"
        for part in document.parts
        if part.kind == "text"
    ]
    images = [
        part for part in document.parts if part.kind == "image" and part.image_bytes
    ][:MAX_IMAGES_PER_DOCUMENT]
    try:
        return await _structure_text(
            generation_client, "\n\n".join(blocks), path_str, images=images
        )
    except ModelGatewayError as exc:
        # The model ran out of output budget part-way through. Losing the tail of a
        # document silently is worse than spending more calls, so fall back to the
        # staged path, which structures it in smaller pieces.
        logger.warning(
            "inline structuring of %s hit the output limit (%s); retrying in stages",
            path_str,
            exc,
        )
        return await _parse_staged(document, generation_client, path_str)


async def _parse_staged(
    document: "Document", generation_client: GenerationClient, path_str: str
) -> str:
    """Structure the document piece by piece when it cannot fit in one request."""
    partials: list[str] = []
    for part in document.parts:
        if part.kind != "text":
            continue
        text = part.text or ""
        slices = max(1, -(-len(text) // INLINE_BUDGET_CHARS))
        for index in range(slices):
            window = text[index * INLINE_BUDGET_CHARS : (index + 1) * INLINE_BUDGET_CHARS]
            partials.append(
                await _structure_text(
                    generation_client,
                    window,
                    f"{path_str} (part {part.part_id}, slice {index + 1}/{slices})",
                )
            )
    combined = "\n\n".join(partials)
    if len(combined) <= INLINE_BUDGET_CHARS:
        return await _structure_text(generation_client, combined, path_str)
    # Refuse to silently truncate: a document too large to unify keeps its parts.
    logger.warning(
        "%s is too large to unify in one request (%d chars); returning the per-part "
        "structure",
        path_str,
        len(combined),
    )
    return combined


async def _structure_text(
    generation_client: GenerationClient,
    text: str,
    source_label: str,
    *,
    images: list[Any] | None = None,
) -> str:
    """Ask for Markdown with true heading depth over the text (and images) supplied."""
    prompt = (
        f"Reconstruct the structure of this document: {source_label}\n\n"
        "Output Markdown. Two rules matter most:\n"
        "1. NESTING -> HEADING DEPTH. The heading markers must reflect how deeply an "
        "entry is nested. The document title is `#`. The outermost entries are `##`. "
        "Entries nested one level inside them are `###`, then `####`, and so on. "
        "The real problem to avoid is putting EVERY heading at `#`: that flattens the "
        "document and destroys its hierarchy, which is worse than making a borderline "
        "judgement call. Chinese official documents express nesting through numbering "
        "(一、/（一）/1./（1）/①), through indentation, or through both - use them as "
        "the nesting signal.\n"
        "2. NOTHING MAY BE LOST. Body text verbatim: do not paraphrase, summarise, "
        "reorder, merge, split or drop anything. Tables become Markdown tables.\n"
        "   IMAGES ARE CONTENT, NOT DECORATION. For every image you are given, transcribe "
        "what it shows as body text under the heading it belongs to: interface labels, "
        "form fields, menu items, the text of a diagram or a form. A guide made of "
        "screenshots must come back as a guide with text, not as a list of titles.\n\n"
        "Worked example of the mapping for a typical Chinese official notice - how deep "
        "each entry sits decides its marker:\n"
        "  document title            -> #\n"
        "  a top-level chapter       -> ##\n"
        "  a sub-item of a chapter   -> ###\n"
        "  an item under that        -> ####\n"
        "The letter or numeral used is irrelevant; what matters is that an entry nested "
        "inside another gets one more `#` than its parent.\n\n"
        f"--- document ---\n{text}\n--- end ---"
    )
    content_parts: Any = prompt
    if images:
        content_parts = [{"type": "text", "text": prompt}]
        for index, part in enumerate(images, start=1):
            content_parts.append(
                {"type": "text", "text": f"[image {index}: {part.label}]"}
            )
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            f"data:{part.mime};base64,"
                            f"{base64.b64encode(part.image_bytes or b'').decode('ascii')}"
                        )
                    },
                }
            )
    data = await generation_client.generate(
        [{"role": "user", "content": content_parts}],
        temperature=0,
        max_tokens=STRUCTURE_MAX_TOKENS,
    )
    content = str(data.get("content") or "").strip()
    if not content:
        raise ModelOutputError(
            f"the document parser returned nothing for {source_label}"
        )
    return _strip_fence(content)


def capabilities_allow_tool_loop() -> bool:
    """Whether the tool-driven strategy may be used.

    Off by default: measured providers can pass a capability probe and still emit tool
    syntax as text at the real prompt, and a strategy that depends on a protocol the
    provider only sometimes honours is not a general strategy.
    """
    return os.getenv("RAG_AGENTIC_TOOL_LOOP_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class AgenticDocumentParser:
    """Ingest-facing wrapper: probe capabilities once, parse every document adaptively."""

    def __init__(
        self,
        generation_client: GenerationClient,
        access: "DocumentAccess",
        *,
        probe_vision: bool = True,
        probe_tools: bool = True,
    ) -> None:
        self.generation_client = generation_client
        self.access = access
        self._probe_vision = probe_vision
        self._probe_tools = probe_tools
        self._capabilities: "ModelCapabilities | None" = None

    async def capabilities(self) -> "ModelCapabilities":
        if self._capabilities is None:
            self._capabilities = await probe_capabilities(
                self.generation_client,
                vision=self._probe_vision,
                tools=self._probe_tools,
            )
        return self._capabilities

    async def parse(self, path: Path) -> str:
        """Parse adaptively, or raise ``ParsingError`` when no strategy can apply.

        Indeterminate capabilities make every downstream choice a guess, so the caller
        is told and falls back to the deterministic parser instead of guessing.
        """
        capabilities = await self.capabilities()
        if capabilities.supports_tools is None and capabilities.supports_vision is None:
            raise ParsingError(
                "model capabilities could not be determined, so adaptive parsing "
                "cannot choose a strategy"
            )
        return await parse_adaptive(
            str(path), self.generation_client, self.access, capabilities
        )
