from langchain_aws import ChatBedrockConverse
from ragas.llms import LangchainLLMWrapper

from src.config.settings import AWS_REGION, BEDROCK_EVALUATION_MODEL


def create_ragas_llm():
    """
    Returns the LLM wrapper used by RAGAS for evaluation.
    """

    llm = ChatBedrockConverse(
        model=BEDROCK_EVALUATION_MODEL,
        region_name=AWS_REGION,
        temperature=0,
    )

    return LangchainLLMWrapper(llm)
