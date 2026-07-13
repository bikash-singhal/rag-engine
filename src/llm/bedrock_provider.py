from src.llm.base import LLMProvider


class BedrockProvider(LLMProvider):
    """
    AWS Bedrock implementation.

    Will be implemented next.
    """

    def __init__(
        self,
        model: str,
    ) -> None:

        self.model = model

    def generate(
        self,
        prompt: str,
    ) -> str:

        raise NotImplementedError(
            "Bedrock provider not implemented yet."
        )