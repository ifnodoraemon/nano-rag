"""Probe what a model can actually do, instead of assuming it.

Every gateway is different, and the difference is invisible until a request fails in a
confusing way. This project has been bitten twice already: a gateway that accepts
``response_format`` and ignores it, and one that emits tool calls as DSML text inside
``content`` rather than as native ``tool_calls``. Both looked like "the model returned
nothing", sending the investigation to the wrong place.

So capabilities are MEASURED once per process and cached, never inferred from a model
name or a vendor:

- ``supports_tools``   : does the provider return structured ``tool_calls``?
- ``supports_vision``  : does the provider accept image input and report seeing it?
- ``json_schema_effective`` : does ``response_format`` actually constrain output?

A probe costs one tiny request per capability and is cached for the process lifetime.
When a probe cannot answer (network blip), the capability is reported as unknown and the
caller falls back to the most broadly compatible strategy rather than assuming a yes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.model_client.generation import GenerationClient

logger = logging.getLogger(__name__)

# A 1x1 PNG. Its only purpose is to see whether the provider accepts an image part at
# all and whether the answer shows the model actually looked at it.
_PROBE_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGA"
    "hKmMIQAAAABJRU5ErkJggg=="
)
_PROBE_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "probe_echo",
            "description": "Echo a string back.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        },
    }
]


@dataclass
class ModelCapabilities:
    """What the configured model can do. ``None`` means "not determined"."""

    supports_tools: bool | None = None
    supports_vision: bool | None = None
    json_schema_effective: bool | None = None
    notes: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "supports_tools": self.supports_tools,
            "supports_vision": self.supports_vision,
            "json_schema_effective": self.json_schema_effective,
            "notes": dict(self.notes),
        }

    @property
    def can_read_images(self) -> bool:
        """Only a confirmed-capable model is allowed to read images."""
        return self.supports_vision is True


async def probe_capabilities(
    generation_client: GenerationClient,
    *,
    vision: bool = True,
    tools: bool = True,
    json_schema: bool = False,
) -> ModelCapabilities:
    """Measure capabilities with minimal requests. Failures become ``None``."""
    capabilities = ModelCapabilities()

    if tools:
        try:
            data = await generation_client.generate(
                [
                    {
                        "role": "user",
                        "content": "Call probe_echo with value 'ok'. Do nothing else.",
                    }
                ],
                temperature=0,
                tools=_PROBE_TOOL,
                tool_choice="auto",
            )
            capabilities.supports_tools = bool(data.get("tool_calls"))
            if not capabilities.supports_tools:
                capabilities.notes["tools"] = (
                    "provider did not return structured tool_calls; tool-driven parsing "
                    "will fall back to injecting the content directly"
                )
        except Exception as exc:  # noqa: BLE001 - a probe must never crash the caller
            capabilities.notes["tools"] = f"probe failed: {type(exc).__name__}: {exc}"

    if vision:
        try:
            data = await generation_client.generate(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Reply with exactly one word. If you can see the "
                                    "image, reply 'visible'. If you cannot, reply 'none'."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{_PROBE_PNG}"
                                },
                            },
                        ],
                    }
                ],
                temperature=0,
            )
            reply = str(data.get("content") or "").strip().lower()
            capabilities.supports_vision = "visible" in reply
            if not capabilities.supports_vision:
                capabilities.notes["vision"] = (
                    f"model did not confirm seeing the image (replied {reply[:60]!r}); "
                    "PDF pages and pictures cannot be parsed by this model"
                )
        except Exception as exc:  # noqa: BLE001
            capabilities.notes["vision"] = (
                f"probe failed: {type(exc).__name__}: {exc}"
            )

    if json_schema:
        try:
            data = await generation_client.generate(
                [{"role": "user", "content": "Return JSON with a single key 'ok'."}],
                temperature=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "probe",
                        "schema": {
                            "type": "object",
                            "properties": {"ok": {"type": "boolean"}},
                            "required": ["ok"],
                        },
                    },
                },
            )
            content = str(data.get("content") or "").strip()
            capabilities.json_schema_effective = content.startswith("{") and "ok" in content
        except Exception as exc:  # noqa: BLE001
            capabilities.notes["json_schema"] = (
                f"probe failed: {type(exc).__name__}: {exc}"
            )

    logger.info("model capabilities: %s", capabilities.as_dict())
    return capabilities
