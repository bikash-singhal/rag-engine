from collections.abc import Iterator

from openai import OpenAI

from src.config.settings import OPENAI_API_KEY
from src.llm.base import LLMProvider


class OpenAIProvider(LLMProvider):

    def __init__(self, model: str) -> None:

        if OPENAI_API_KEY is None:
            raise RuntimeError("OPENAI_API_KEY is required to use OpenAIProvider.")

        self.client = OpenAI(api_key=OPENAI_API_KEY)

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

        content = response.choices[0].message.content

        if content is None:
            raise RuntimeError("OpenAI did not generate a response.")

        return content

    def stream(
        self,
        prompt: str,
    ) -> Iterator[str]:

        # TODO: Implement streaming.
        raise NotImplementedError()
