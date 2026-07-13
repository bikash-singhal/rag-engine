from openai import OpenAI

from src.llm.openai_provider import OpenAIProvider
from src.llm.bedrock_provider import BedrockProvider


def get_provider(provider_name: str, model: str,):    
    """
    Factory method that returns the configured LLM provider.
    """

    provider = provider_name.lower()
    
    if provider == "openai":
        client = OpenAI()

        return OpenAIProvider(
            client=client,
            model=model,
        )

    elif provider == "bedrock":

        return BedrockProvider(
            model=model,
        )

    else:
        raise ValueError(
            f"Unsupported LLM provider: {LLM_PROVIDER}"
        )