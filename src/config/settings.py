import os

from dotenv import load_dotenv

load_dotenv()


def required_env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise ValueError(f"Missing or empty required environment variable: {name}")

    return value


def optional_env(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value else None


# AWS
AWS_PROFILE: str = required_env("AWS_PROFILE")
AWS_REGION: str = required_env("AWS_REGION")


# Bedrock
BEDROCK_GENERATION_MODEL: str = required_env("BEDROCK_GENERATION_MODEL")
BEDROCK_EVALUATION_MODEL: str = required_env("BEDROCK_EVALUATION_MODEL")


# OpenAI
OPENAI_API_KEY: str | None = optional_env("OPENAI_API_KEY")


# Embedding and reranking
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


# Chunking
CHUNK_SIZE: int = 400
CHUNK_OVERLAP: int = 100


# Retrieval
DENSE_WEIGHT: float = 0.7
BM25_WEIGHT: float = 0.3

RETRIEVAL_TOP_K: int = 30
MAX_RERANK_CANDIDATES: int = 20
FINAL_TOP_K: int = 5

NUM_MULTI_QUERIES: int = 4
QUERY_REWRITE_HISTORY: int = 4
