from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are a grounded question-answering assistant. Answer only from the provided context and cite the supporting sources.
If the context is insufficient, say so explicitly.
When you cite evidence, use the provided citation labels such as [C1] or [C2]."""

# Media types that can be sent inline to a vision-capable LLM.
_VISION_MIME_PREFIXES = ("image/",)


class PromptBuilder:
    def __init__(self, prompts: dict[str, object]) -> None:
        self.system_prompt = prompts.get("chat", {}).get(
            "system", DEFAULT_SYSTEM_PROMPT
        )
        self._media_inline_max_bytes = int(
            os.getenv("RAG_PROMPT_INLINE_MEDIA_MAX_BYTES", str(10 * 1024 * 1024))
        )

    def build_messages(
        self, query: str, contexts: list[dict[str, object]]
    ) -> list[dict[str, Any]]:
        conflict_notice = ""
        if any(item.get("wiki_status") == "conflicting" for item in contexts):
            conflict_notice = (
                "Warning: some retrieved evidence is marked as conflicting, which means the sources may disagree. "
                "You must explicitly describe the conflict and avoid giving an overly certain conclusion.\n\n"
            )
        context_text = self._render_evidence_sections(contexts)
        instruction = (
            f"Question: {query}\n\n"
            f"{conflict_notice}"
            f"Available context:\n{context_text}\n\n"
            "Answer using only the context above and cite the evidence with the provided labels, for example [C1].\n\n"
            "Return the result in this format:\n"
            "Final Answer:\n"
            "<your answer with citations>\n\n"
            "Supporting Claims:\n"
            "- [factual|conditional|conflict|insufficiency] <claim 1> [C#]\n"
            "- [factual|conditional|conflict|insufficiency] <claim 2> [C#]\n"
            "If there are no strong supporting claims, return `- None`."
        )

        media_parts = list(self._collect_media_parts(contexts))
        if not media_parts:
            return [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": instruction},
            ]
        user_parts: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
        user_parts.extend(media_parts)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_parts},
        ]

    def _collect_media_parts(
        self, contexts: list[dict[str, object]]
    ) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        for item in contexts:
            modality = item.get("modality")
            mime = str(item.get("mime_type") or "")
            media_uri = item.get("media_uri")
            if modality != "image" or not media_uri:
                continue
            if not any(mime.startswith(prefix) for prefix in _VISION_MIME_PREFIXES):
                continue
            data = self._read_media(str(media_uri))
            if data is None:
                continue
            b64 = base64.b64encode(data).decode("ascii")
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }
            )
        return parts

    def _read_media(self, media_uri: str) -> bytes | None:
        path = Path(media_uri)
        candidates = [path]
        if not path.is_absolute():
            cwd = Path.cwd()
            candidates.append(cwd / path)
            candidates.append(cwd.parent / path)
        for candidate in candidates:
            try:
                if candidate.is_file():
                    size = candidate.stat().st_size
                    if size > self._media_inline_max_bytes:
                        logger.warning(
                            "media %s exceeds inline limit (%d bytes); skipping",
                            candidate,
                            size,
                        )
                        return None
                    return candidate.read_bytes()
            except OSError:
                continue
        logger.info(
            "image-grounded generation: media bytes unavailable for %s; "
            "falling back to text-only prompt",
            media_uri,
        )
        return None

    def _render_evidence_sections(self, contexts: list[dict[str, object]]) -> str:
        role_titles = {
            "primary": "Primary Evidence",
            "supporting": "Supporting Evidence",
            "conflicting": "Conflicting Evidence",
        }
        sections: list[str] = []
        for role in ("primary", "supporting", "conflicting"):
            role_contexts = [
                item for item in contexts if item.get("evidence_role", "supporting") == role
            ]
            if not role_contexts:
                continue
            entries = "\n\n".join(self._render_context(item) for item in role_contexts)
            sections.append(f"## {role_titles[role]}\n{entries}")
        if not sections:
            return ""
        return "\n\n".join(sections)

    def _render_context(self, item: dict[str, object]) -> str:
        modality = item.get("modality") or "text"
        body = item.get("text") or ""
        if modality != "text":
            media_uri = item.get("media_uri") or "n/a"
            mime_type = item.get("mime_type") or "n/a"
            body = f"[{modality} attachment: {media_uri} ({mime_type})]"
        return (
            f"[{item.get('citation_label') or item.get('chunk_id', 'unknown')}] "
            f"(evidence={item.get('evidence_role') or 'supporting'}) "
            f"(modality={modality}) "
            f"(kind={item.get('wiki_kind') or 'raw'}, status={item.get('wiki_status') or 'n/a'}) "
            f"(section={item.get('section_path') or 'n/a'}, doc_type={item.get('doc_type') or 'n/a'}) "
            f"(effective_date={item.get('effective_date') or 'n/a'}, version={item.get('version') or 'n/a'}, freshness={item.get('freshness_tier') or 'n/a'}) "
            f"{body}"
        )
