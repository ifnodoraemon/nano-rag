from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """你是一个基于证据回答问题的助手。只能根据提供的上下文回答，并引用真正支撑结论的来源。
如果上下文不足以回答，必须明确说明信息不足。
引用证据时使用提供的标签，例如 [C1] 或 [C2]。"""

# Media types that can be sent inline to a vision-capable LLM.
_VISION_MIME_PREFIXES = ("image/",)
DEFAULT_PROMPT_CONTEXT_MAX_CHARS = 48000
DEFAULT_PROMPT_CONTEXT_ITEM_MAX_CHARS = 6000


class PromptBuilder:
    def __init__(self, prompts: dict[str, object]) -> None:
        self.system_prompt = prompts.get("chat", {}).get(
            "system", DEFAULT_SYSTEM_PROMPT
        )
        self._media_inline_max_bytes = int(
            os.getenv("RAG_PROMPT_INLINE_MEDIA_MAX_BYTES", str(10 * 1024 * 1024))
        )
        self._context_max_chars = int(
            os.getenv("RAG_PROMPT_CONTEXT_MAX_CHARS", str(DEFAULT_PROMPT_CONTEXT_MAX_CHARS))
        )
        self._context_item_max_chars = int(
            os.getenv(
                "RAG_PROMPT_CONTEXT_ITEM_MAX_CHARS",
                str(DEFAULT_PROMPT_CONTEXT_ITEM_MAX_CHARS),
            )
        )

    def build_messages(
        self,
        query: str,
        contexts: list[dict[str, object]],
        agent_state: dict[str, object] | None = None,
    ) -> list[dict[str, Any]]:
        conflict_notice = ""
        if any(item.get("wiki_status") == "conflicting" for item in contexts):
            conflict_notice = (
                "Warning: some retrieved evidence is marked as conflicting, which means the sources may disagree. "
                "You must explicitly describe the conflict and avoid giving an overly certain conclusion.\n\n"
            )
        context_text = self._render_evidence_sections(contexts)
        question_input = json.dumps({"question": query}, ensure_ascii=False)
        instruction = (
            f"Question input JSON: {question_input}\n\n"
            f"{conflict_notice}"
            f"{self._render_agent_state(agent_state)}"
            f"Available context:\n{context_text}\n\n"
            "只根据上面的上下文回答，并使用提供的标签引用证据，例如 [C1]。\n"
            "【防幻觉与精简指令】：回答必须极度简练、直击要害。如果用户询问特定术语（如“术语A”）、文件条款或专有名词，但在上下文中找不到完全吻合的具体名称，请直接回答：“文档中未包含关于【在此填入用户询问的具体名称】的相关信息。” 绝对不要把文档里的其他名词强行解释为用户询问的词。\n"
            "不要复述无关背景，不要总结全篇文档内容。绝对禁止使用类似“现有文档仅涉及XXX”的话术来凑字数。\n"
            "优先引用同时包含问题核心实体和答案值的最小证据。不要引用只包含单位、表头、残缺行或泛化说明的片段，除非它们是唯一可用证据。\n"
            "如果单条表格行已经同时包含问题中的实体、限定条件和答案值，直接基于该行回答；不要因为相邻片段不完整而拒答。\n\n"
            "按以下格式返回：\n"
            "Final Answer:\n"
            "<带引用的极简答案>\n\n"
            "Supporting Claims:\n"
            "- [factual|conditional|conflict|insufficiency] <证据支撑点 1> [C#]\n"
            "- [factual|conditional|conflict|insufficiency] <证据支撑点 2> [C#]\n"
            "如果没有强支撑点，返回 `- None`。"
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

    def _render_agent_state(self, agent_state: dict[str, object] | None) -> str:
        if not agent_state:
            return ""
        verification = agent_state.get("verification")
        verification_text = verification if isinstance(verification, dict) else {}
        subqueries = [
            str(item)
            for item in agent_state.get("subqueries", [])
            if str(item).strip()
        ]
        retrieval_queries = [
            str(item)
            for item in agent_state.get("retrieval_queries", [])
            if str(item).strip()
        ]
        graph_expanded = [
            str(item)
            for item in agent_state.get("graph_expanded_node_ids", [])
            if str(item).strip()
        ]
        missing_terms = [
            str(item)
            for item in verification_text.get("missing_terms", [])
            if str(item).strip()
        ]
        return (
            "Agent evidence check:\n"
            f"- subqueries: {'; '.join(subqueries) or 'n/a'}\n"
            f"- retrieval_queries: {'; '.join(retrieval_queries) or 'n/a'}\n"
            f"- graph_expanded_node_ids: {', '.join(graph_expanded) or 'none'}\n"
            f"- evidence_sufficient: {verification_text.get('sufficient')}\n"
            f"- missing_terms: {', '.join(missing_terms) or 'none'}\n"
            f"{self._render_evidence_plan(agent_state.get('evidence_plan'))}"
            "如果 evidence_sufficient 为 false，最终答案必须明确说明现有证据不足，并列出还缺少哪些信息。\n\n"
        )

    def _render_evidence_plan(self, value: object) -> str:
        if not isinstance(value, dict):
            return ""
        strategy = value.get("answer_strategy") or "direct"
        primary = [
            str(item)
            for item in value.get("primary_evidence", [])
            if str(item).strip()
        ]
        conditions = [
            str(item)
            for item in value.get("conditions", [])
            if str(item).strip()
        ]
        relations = [
            item
            for item in value.get("relations", [])
            if isinstance(item, dict)
        ]
        outline = [
            str(item)
            for item in value.get("outline", [])
            if str(item).strip()
        ]
        rendered_relations = []
        for relation in relations[:6]:
            rendered_relations.append(
                f"{relation.get('source')} {relation.get('relation')} {relation.get('target')}"
            )
        return (
            "- evidence_plan_strategy: "
            f"{strategy}\n"
            f"- evidence_plan_primary: {', '.join(primary) or 'none'}\n"
            f"- evidence_plan_conditions: {'; '.join(conditions[:4]) or 'none'}\n"
            f"- evidence_plan_relations: {'; '.join(rendered_relations) or 'none'}\n"
            f"- evidence_plan_outline: {'; '.join(outline[:5]) or 'none'}\n"
            "必须遵循 evidence_plan：有条件就给条件化结论，有 contradictions/冲突关系就明确说明冲突，不能只挑单边证据。\n"
        )

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
            entries = []
            for item in role_contexts:
                rendered = self._render_context(item)
                projected = "\n\n".join([*sections, f"## {role_titles[role]}\n", *entries, rendered])
                if len(projected) > self._context_max_chars:
                    logger.warning(
                        "prompt context budget reached at %d chars; dropping remaining contexts",
                        self._context_max_chars,
                    )
                    break
                entries.append(rendered)
            if entries:
                sections.append(f"## {role_titles[role]}\n" + "\n\n".join(entries))
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
        elif isinstance(body, str) and len(body) > self._context_item_max_chars:
            body = body[: self._context_item_max_chars - 3].rstrip() + "..."
        return (
            f"[{item.get('citation_label') or item.get('chunk_id', 'unknown')}] "
            f"(evidence={item.get('evidence_role') or 'supporting'}) "
            f"(modality={modality}) "
            f"(kind={item.get('wiki_kind') or 'raw'}, status={item.get('wiki_status') or 'n/a'}) "
            f"(section={item.get('section_path') or 'n/a'}, doc_type={item.get('doc_type') or 'n/a'}) "
            f"(effective_date={item.get('effective_date') or 'n/a'}, version={item.get('version') or 'n/a'}, freshness={item.get('freshness_tier') or 'n/a'}) "
            f"(claim_role={item.get('claim_role') or 'n/a'}, certainty={item.get('certainty') or 'n/a'}, scope={item.get('claim_scope') or 'n/a'}) "
            f"{body}"
        )
