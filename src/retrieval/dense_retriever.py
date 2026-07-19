from src.core.models import SearchResult
from src.embeddings.embedder import Embedder
from src.retrieval.retriever import Retriever
from src.utils.logger import get_logger
from src.vectorstores.faiss_store import FAISSVectorStore

logger = get_logger(__name__)


class DenseRetriever(Retriever):
    """
    Dense vector retrieval using embeddings and FAISS.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: FAISSVectorStore,
    ) -> None:

        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:

        logger.debug("Dense retrieval started.")
        query_embedding = self.embedder.embed_query(query)

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        logger.debug(
            "Dense retrieval returned %d results.",
            len(results),
        )

        return results
