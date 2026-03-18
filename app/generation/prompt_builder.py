from __future__ import annotations


class PromptBuilder:
    def __init__(self, prompts: dict) -> None:
        self.system_prompt = prompts["chat"]["system"]

    def build_messages(self, query: str, contexts: list[dict[str, object]]) -> list[dict[str, str]]:
        context_text = "\n\n".join(
            f"[{item['chunk_id']}] {item['text']}" for item in contexts
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"问题：{query}\n\n可用上下文：\n{context_text}\n\n请基于上下文作答并给出引用。",
            },
        ]
