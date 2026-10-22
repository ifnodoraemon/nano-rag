"""Strategy selection is driven by the DOCUMENT and the MODEL, not by file format."""
from __future__ import annotations

from app.ingestion.agentic_parser import choose_strategy
from app.ingestion.document_access import Document, Part
from app.model_client.capabilities import ModelCapabilities


def _doc(text_chars: int = 100, images: int = 0) -> Document:
    parts = [Part(part_id="p1", kind="text", text="x" * text_chars)]
    parts += [Part(part_id=f"pg{i}", kind="image", extra={"page": i}) for i in range(1, images + 1)]
    return Document(kind="mixed", parts=parts)


def test_a_visionless_model_refuses_image_parts_instead_of_faking_text():
    """No OCR fallback: refusing is the honest answer."""
    strategy, reason = choose_strategy(_doc(images=3), ModelCapabilities(supports_vision=False))
    assert strategy == "unsupported"
    assert "vision" in reason


def test_an_unmeasured_model_is_not_assumed_to_see_images():
    # supports_vision is None after a failed probe -> must not be treated as capable.
    strategy, _ = choose_strategy(_doc(images=1), ModelCapabilities(supports_vision=None))
    assert strategy == "unsupported"


def test_text_only_document_with_unknown_vision_is_fine():
    strategy, _ = choose_strategy(_doc(images=0), ModelCapabilities(supports_vision=None))
    assert strategy in {"inline", "tool_loop"}


def test_tool_calling_is_opt_in_not_the_default(monkeypatch):
    """A provider can pass a probe and still emit tool syntax as plain text.

    Inline/staged does not depend on that protocol, so it is the baseline; the loop is
    enabled per deployment.
    """
    monkeypatch.delenv("RAG_AGENTIC_TOOL_LOOP_ENABLED", raising=False)
    strategy, _ = choose_strategy(_doc(10), ModelCapabilities(supports_tools=True))
    assert strategy == "inline"

    monkeypatch.setenv("RAG_AGENTIC_TOOL_LOOP_ENABLED", "true")
    strategy, _ = choose_strategy(_doc(10), ModelCapabilities(supports_tools=True))
    assert strategy == "tool_loop"


def test_small_document_without_tools_is_inlined():
    strategy, reason = choose_strategy(_doc(5_000), ModelCapabilities(supports_tools=False))
    assert strategy == "inline"
    assert "fit in one request" in reason


def test_large_document_without_tools_is_staged():
    strategy, _ = choose_strategy(_doc(200_000), ModelCapabilities(supports_tools=False))
    assert strategy == "staged"


def test_a_vision_capable_model_is_allowed_to_process_images():
    strategy, reason = choose_strategy(
        _doc(1_000, images=2), ModelCapabilities(supports_vision=True)
    )
    # No tool loop by default, but images are no longer a refusal.
    assert strategy in {"inline", "staged"}
    assert "vision" not in reason
