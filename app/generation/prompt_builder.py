from __future__ import annotations

DEFAULT_SYSTEM_PROMPT = """You are a grounded question-answering assistant. Answer only from the provided context and cite the supporting sources.
If the context is insufficient, say so explicitly.
When you cite evidence, use the provided citation labels such as [C1] or [C2]."""


class PromptBuilder:
    def __init__(self, prompts: dict[str, object]) -> None:
        self.system_prompt = prompts.get("chat", {}).get(
            "system", DEFAULT_SYSTEM_PROMPT
        )

    def build_messages(
        self, query: str, contexts: list[dict[str, object]]
    ) -> list[dict[str, str]]:
        conflict_notice = ""
        if any(item.get("wiki_status") == "conflicting" for item in contexts):
            conflict_notice = (
                "Warning: some retrieved evidence is marked as conflicting, which means the sources may disagree. "
                "You must explicitly describe the conflict and avoid giving an overly certain conclusion.\n\n"
            )
        context_text = self._render_evidence_sections(contexts)
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
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
                ),
            },
        ]

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
        return (
            f"[{item.get('citation_label') or item.get('chunk_id', 'unknown')}] "
            f"(evidence={item.get('evidence_role') or 'supporting'}) "
            f"(kind={item.get('wiki_kind') or 'raw'}, status={item.get('wiki_status') or 'n/a'}) "
            f"(section={item.get('section_path') or 'n/a'}, doc_type={item.get('doc_type') or 'n/a'}) "
            f"(effective_date={item.get('effective_date') or 'n/a'}, version={item.get('version') or 'n/a'}, freshness={item.get('freshness_tier') or 'n/a'}) "
            f"{item.get('text', '')}"
        )
