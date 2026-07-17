from abc import ABC, abstractmethod

from src.core.models import SearchResult


class Reranker(ABC):
    """
    Base interface for rerankers.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Reorders retrieved results according to relevance.
        """
        ...
