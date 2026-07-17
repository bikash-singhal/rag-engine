from sentence_transformers import CrossEncoder

from src.config.settings import RERANKER_MODEL
from src.core.models import SearchResult
from src.reranking.reranker import Reranker


class CrossEncoderReranker(Reranker):

    def __init__(self) -> None:

        if not RERANKER_MODEL:
            raise RuntimeError("RERANKER_MODEL is not configured.")

        self.model = CrossEncoder(RERANKER_MODEL)

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:

        if not results:
            return []

        pairs = [(query, result.chunk.text) for result in results]

        scores = self.model.predict(pairs)

        reranked_results = []

        reranked_results = [
            SearchResult(
                chunk=result.chunk,
                score=float(score),
            )
            for result, score in zip(results, scores)
        ]

        reranked_results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return reranked_results[:top_k]
