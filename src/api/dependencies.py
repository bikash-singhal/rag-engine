from functools import lru_cache
from pathlib import Path

from src.api.job_manager import JobManager
from src.app.rag_engine import RAGEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache
def get_rag_engine() -> RAGEngine:

    engine = RAGEngine()

    index_dir = Path("data/indexes/default")

    faiss = index_dir / "faiss.index"
    embedded_chunks = index_dir / "embedded_chunks.pkl"

    if faiss.exists() and embedded_chunks.exists():
        logger.info("Loading default index from %s", index_dir)

        try:
            engine.load_index(index_dir)
            logger.info("Default index loaded successfully.")
        except Exception:
            logger.exception("Failed to load default index.")
    else:
        logger.info("No default index found. Start by ingesting a document.")

    return engine


@lru_cache
def get_job_manager() -> JobManager:
    return JobManager()
