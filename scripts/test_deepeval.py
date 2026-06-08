import asyncio
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from openai import AsyncOpenAI
import os
import sys

class CustomOpenAI(DeepEvalBaseLLM):
    def __init__(self):
        self.model = "gemini-3.1-pro-preview"
        self.client = AsyncOpenAI(
            base_url=os.getenv("RAG_RAGAS_LIB_LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
            api_key=os.getenv("GEMINI_API_KEY")
        )

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
        return loop.run_until_complete(self.a_generate(prompt))

    async def a_generate(self, prompt: str) -> str:
        res = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content

    def get_model_name(self):
        return self.model

def main():
    model = CustomOpenAI()
    test_case = LLMTestCase(
        input="病假超过几天需要医院证明？",
        actual_output="根据公司病假制度，病假超过 3 天需要提供医院证明。",
        retrieval_context=["病假超过 3 天需要提供医院证明。"]
    )
    metric = FaithfulnessMetric(threshold=0.5, model=model)
    metric.measure(test_case)
    print("Faithfulness Score:", metric.score)
    print("Reason:", metric.reason)

if __name__ == "__main__":
    main()
