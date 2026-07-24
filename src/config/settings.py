# settings.py

import os

from dotenv import load_dotenv

load_dotenv()


def required_env(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


AWS_PROFILE: str = required_env("AWS_PROFILE")
AWS_REGION: str = required_env("AWS_REGION")

BEDROCK_GENERATION_MODEL: str = required_env("BEDROCK_GENERATION_MODEL")
BEDROCK_EVALUATION_MODEL: str = required_env("BEDROCK_EVALUATION_MODEL")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "BAAI/bge-reranker-base"

LLM_PROVIDER = "openai"
LLM_MODEL = "gpt-4.1-mini"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 100

TOP_K = 5

DENSE_WEIGHT = 0.7
BM25_WEIGHT = 0.3

RETRIEVAL_TOP_K = 30
MAX_RERANK_CANDIDATES = 20
FINAL_TOP_K = 5
NUM_MULTI_QUERIES = 4
QUERY_REWRITE_HISTORY = 4
