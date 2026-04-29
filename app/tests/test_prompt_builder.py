import base64
from pathlib import Path

from app.generation.prompt_builder import PromptBuilder


def test_prompt_builder_warns_when_contexts_are_conflicting() -> None:
    builder = PromptBuilder(prompts={})

    messages = builder.build_messages(
        "Is PTO carryover allowed?",
        [
            {
                "chunk_id": "wiki:topic:leave-policy",
                "citation_label": "C1",
                "text": "PTO carryover is allowed up to 5 days.",
                "wiki_kind": "topic",
                "wiki_status": "conflicting",
                "evidence_role": "conflicting",
            }
        ],
    )

    assert "Warning: some retrieved evidence is marked as conflicting" in messages[1]["content"]
    assert "## Conflicting Evidence" in messages[1]["content"]
    assert "[C1]" in messages[1]["content"]
    assert "evidence=conflicting" in messages[1]["content"]
    assert "kind=topic" in messages[1]["content"]
    assert "status=conflicting" in messages[1]["content"]


def test_prompt_builder_includes_section_metadata() -> None:
    builder = PromptBuilder(prompts={})

    messages = builder.build_messages(
        "What is the carryover policy?",
        [
            {
                "chunk_id": "c1",
                "citation_label": "C1",
                "text": "Leave Policy section. Carryover is allowed up to 5 days.",
                "section_path": ["Policy", "Leave Policy"],
                "doc_type": "policy",
                "effective_date": "2026-01-15",
                "version": "v2.1",
                "freshness_tier": "primary",
                "evidence_role": "primary",
            }
        ],
    )

    assert "## Primary Evidence" in messages[1]["content"]
    assert "[C1]" in messages[1]["content"]
    assert "section=['Policy', 'Leave Policy']" in messages[1]["content"]
    assert "doc_type=policy" in messages[1]["content"]
    assert "effective_date=2026-01-15" in messages[1]["content"]
    assert "version=v2.1" in messages[1]["content"]
    assert "freshness=primary" in messages[1]["content"]


def test_prompt_builder_separates_primary_supporting_and_conflicting_evidence() -> None:
    builder = PromptBuilder(prompts={})

    messages = builder.build_messages(
        "What is the carryover policy?",
        [
            {
                "chunk_id": "primary-1",
                "citation_label": "C1",
                "text": "Current policy allows carryover up to 5 days.",
                "evidence_role": "primary",
            },
            {
                "chunk_id": "supporting-1",
                "citation_label": "C2",
                "text": "Manager approval is required.",
                "evidence_role": "supporting",
            },
            {
                "chunk_id": "conflict-1",
                "citation_label": "C3",
                "text": "Older policy says carryover is not allowed.",
                "wiki_status": "conflicting",
                "evidence_role": "conflicting",
            },
        ],
    )

    content = messages[1]["content"]
    assert "## Primary Evidence" in content
    assert "## Supporting Evidence" in content
    assert "## Conflicting Evidence" in content
    assert "[C1]" in content
    assert "[C2]" in content
    assert "[C3]" in content
    assert "Final Answer:" in content
    assert "Supporting Claims:" in content
    assert "[factual|conditional|conflict|insufficiency]" in content
    assert content.index("## Primary Evidence") < content.index("## Supporting Evidence")
    assert content.index("## Supporting Evidence") < content.index("## Conflicting Evidence")


def test_prompt_builder_inlines_image_chunks(tmp_path: Path) -> None:
    image_bytes = b"\x89PNGfake"
    image_path = tmp_path / "logo.png"
    image_path.write_bytes(image_bytes)

    builder = PromptBuilder(prompts={})
    messages = builder.build_messages(
        "Identify the company in this image.",
        [
            {
                "chunk_id": "doc-img:0",
                "citation_label": "C1",
                "text": "",
                "modality": "image",
                "media_uri": str(image_path),
                "mime_type": "image/png",
                "evidence_role": "primary",
            }
        ],
    )

    user = messages[1]
    assert isinstance(user["content"], list)
    text_part, image_part = user["content"]
    assert text_part["type"] == "text"
    assert "modality=image" in text_part["text"]
    assert image_part["type"] == "image_url"
    expected_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode()
    assert image_part["image_url"]["url"] == expected_url


def test_prompt_builder_text_only_when_image_bytes_unavailable() -> None:
    builder = PromptBuilder(prompts={})
    messages = builder.build_messages(
        "What does the chart show?",
        [
            {
                "chunk_id": "doc-img:0",
                "citation_label": "C1",
                "text": "",
                "modality": "image",
                "media_uri": "/nonexistent/path/image.png",
                "mime_type": "image/png",
                "evidence_role": "primary",
            }
        ],
    )

    # Bytes unavailable → fall back to plain string content (no image_url part).
    assert isinstance(messages[1]["content"], str)
    assert "modality=image" in messages[1]["content"]


def test_prompt_builder_preserves_text_only_string_content() -> None:
    builder = PromptBuilder(prompts={})
    messages = builder.build_messages(
        "How long is PTO carryover?",
        [
            {
                "chunk_id": "c1",
                "citation_label": "C1",
                "text": "PTO carryover is up to 5 days.",
                "evidence_role": "primary",
            }
        ],
    )

    # Pure-text contexts must keep the historical string-content shape so
    # all existing generation client paths keep
    # working.
    assert isinstance(messages[1]["content"], str)
