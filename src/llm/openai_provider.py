import os

from openai import OpenAI

from src.llm.base import LLMProvider


class OpenAIProvider(LLMProvider):

    def __init__(self, model: str) -> None:

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.model = model

    def generate(self, prompt: str) -> str:

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        val = response.choices[0].message.content

        if val is None:
            raise RuntimeError("OpenAI did not generate response")

        return val
