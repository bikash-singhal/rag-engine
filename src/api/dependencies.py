from functools import lru_cache

from src.app.rag_engine import RAGEngine


@lru_cache(maxsize=1)
def get_rag_engine() -> RAGEngine:
    """
    Returns a singleton RAG Engine.

    The engine is created only once and reused
    for every request.
    """

    engine = RAGEngine()

    engine.load_or_ingest(
        document_directory="data/raw",
        index_directory="data/indexes/default",
    )

    return engine
