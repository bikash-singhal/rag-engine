from abc import ABC, abstractmethod

from src.config.settings import RETRIEVAL_TOP_K
from src.core.models import SearchResult


class Retriever(ABC):
    """
    Base interface for all retrieval strategies.
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> list[SearchResult]:
        """
        Retrieves the most relevant chunks.
        """
        ...
