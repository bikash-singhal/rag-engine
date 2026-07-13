from openai import OpenAI

from src.llm.base import LLMProvider


class OpenAIProvider(LLMProvider):
    """
    OpenAI implementation of the LLM provider interface.
    """

    def __init__(
        self,
        client: OpenAI,
        model: str,
    ) -> None:

        self.client = client
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

        return response.choices[0].message.content