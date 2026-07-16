from abc import ABC, abstractmethod

from src.core.models import SearchResult


class Retriever(ABC):
    """
    Base interface for all retrieval strategies.
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Retrieves the most relevant chunks.
        """
        ...
