from abc import ABC, abstractmethod

from src.config.settings import MAX_RERANK_CANDIDATES
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
        top_k: int = MAX_RERANK_CANDIDATES,
    ) -> list[SearchResult]:
        """
        Reorders retrieved results according to relevance.
        """
        ...
