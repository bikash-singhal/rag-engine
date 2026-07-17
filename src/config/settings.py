# settings.py

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "BAAI/bge-reranker-base"

LLM_PROVIDER = "openai"
LLM_MODEL = "gpt-4.1-mini"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 100

TOP_K = 5

DENSE_WEIGHT = 0.7
BM25_WEIGHT = 0.3
