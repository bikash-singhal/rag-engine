from src.llm.openai_provider import OpenAIProvider
from src.llm.bedrock_provider import BedrockProvider


def get_provider(
    provider_name: str,
    model: str,
):

    provider = provider_name.lower()

    if provider == "openai":

        return OpenAIProvider(
            model=model,
        )

    elif provider == "bedrock":

        return BedrockProvider(
            model=model,
        )

    raise ValueError(
        f"Unsupported provider: {provider_name}"
    )