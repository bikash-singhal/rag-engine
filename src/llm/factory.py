def get_provider(
    provider_name: str,
    model: str,
):

    provider = provider_name.lower()

    if provider == "openai":
        from src.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(
            model=model,
        )

    elif provider == "bedrock":
        from src.llm.bedrock_provider import BedrockProvider

        return BedrockProvider(
            model=model,
        )

    raise ValueError(f"Unsupported provider: {provider_name}")
