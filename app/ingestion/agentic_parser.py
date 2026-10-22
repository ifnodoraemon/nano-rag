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

import json
import logging
import re
from typing import Any

from app.core.exceptions import ModelOutputError, ParsingError
from app.ingestion.document_access import TOOL_SCHEMAS, DocumentAccess
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
