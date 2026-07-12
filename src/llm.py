from openai import OpenAI


class LLM:
    """
    Wrapper around an LLM provider.

    Currently supports OpenAI-compatible chat models.
    """

    def __init__(
        self,
        client: OpenAI,
        model: str,
    ) -> None:
        self.client = client
        self.model = model

    def generate(
        self,
        prompt: str,
    ) -> str:
        """
        Generates an answer from the LLM.

        Args:
            prompt: Fully formatted prompt.

        Returns:
            Model response as a string.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
        )

        content = response.choices[0].message.content

        return content if content is not None else ""