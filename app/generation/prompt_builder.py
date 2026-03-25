from __future__ import annotations

DEFAULT_SYSTEM_PROMPT = """你是一个专业的问答助手。请基于提供的上下文回答问题，并在答案中标注引用来源。
如果上下文中没有足够的信息回答问题，请明确说明。"""


class PromptBuilder:
    def __init__(self, prompts: dict) -> None:
        self.system_prompt = prompts.get("chat", {}).get(
            "system", DEFAULT_SYSTEM_PROMPT
        )

    def build_messages(
        self, query: str, contexts: list[dict[str, object]]
    ) -> list[dict[str, str]]:
        context_text = "\n\n".join(
            f"[{item.get('chunk_id', 'unknown')}] {item.get('text', '')}"
            for item in contexts
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"问题：{query}\n\n可用上下文：\n{context_text}\n\n请基于上下文作答并给出引用。",
            },
        ]
