from functools import lru_cache

from src.api.job_manager import JobManager
from src.app.rag_engine import RAGEngine
from src.chat.chat_engine import ChatEngine


@lru_cache
def get_rag_engine() -> RAGEngine:

    engine = RAGEngine()

    engine.load_or_ingest(
        document_directory="data/raw",
        index_directory="data/indexes/default",
    )

    return engine


def get_chat_engine() -> ChatEngine:
    return get_rag_engine().chat_engine


@lru_cache
def get_job_manager() -> JobManager:
    return JobManager()
