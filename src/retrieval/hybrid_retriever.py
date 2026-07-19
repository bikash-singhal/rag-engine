import time

from src.config.settings import BM25_WEIGHT, DENSE_WEIGHT, RETRIEVAL_TOP_K
from src.core.models import SearchResult
from src.retrieval.retriever import Retriever
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever(Retriever):

    def __init__(
        self,
        dense_retriever: Retriever,
        bm25_retriever: Retriever,
        dense_weight: float = DENSE_WEIGHT,
        bm25_weight: float = BM25_WEIGHT,
    ) -> None:

        if dense_weight < 0 or bm25_weight < 0:
            raise ValueError("Weights must be non-negative.")

        if dense_weight + bm25_weight == 0:
            raise ValueError("At least one weight must be positive.")

        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    def _retrieve_dense(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> list[SearchResult]:

        return self.dense_retriever.retrieve(
            query=query,
            top_k=top_k,
        )

    def _retrieve_bm25(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> list[SearchResult]:

        return self.bm25_retriever.retrieve(
            query=query,
            top_k=top_k,
        )

    def _minmax_normalize(
        self,
        results: list[SearchResult],
    ) -> dict[int, float]:

        if not results:
            return {}

        scores = [result.score for result in results]

        minimum = min(scores)
        maximum = max(scores)

        if minimum == maximum:

            return {result.chunk.chunk_index: 1.0 for result in results}

        normalized_scores: dict[int, float] = {}

        for result in results:

            normalized_score = (result.score - minimum) / (maximum - minimum)

            normalized_scores[result.chunk.chunk_index] = normalized_score

        return normalized_scores

    def _merge_results(
        self,
        dense_results: list[SearchResult],
        bm25_results: list[SearchResult],
    ) -> list[SearchResult]:

        dense_scores = self._minmax_normalize(dense_results)

        bm25_scores = self._minmax_normalize(bm25_results)

        chunk_lookup = {
            result.chunk.chunk_index: result.chunk
            for result in (dense_results + bm25_results)
        }

        all_chunk_ids = set(dense_scores) | set(bm25_scores)

        merged_results: list[SearchResult] = []

        for chunk_id in all_chunk_ids:

            dense_score = dense_scores.get(
                chunk_id,
                0.0,
            )

            bm25_score = bm25_scores.get(
                chunk_id,
                0.0,
            )

            combined_score = (
                self.dense_weight * dense_score + self.bm25_weight * bm25_score
            )

            merged_results.append(
                SearchResult(
                    chunk=chunk_lookup[chunk_id],
                    score=combined_score,
                )
            )

        merged_results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return merged_results

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:

        start = time.perf_counter()

        dense_results = self._retrieve_dense(query, top_k)

        dense_time = time.perf_counter() - start

        logger.debug(
            "Dense retrieval: %.3f sec (%d results)",
            dense_time,
            len(dense_results),
        )

        start = time.perf_counter()

        bm25_results = self._retrieve_bm25(query, top_k)

        bm25_time = time.perf_counter() - start

        logger.debug(
            "BM25 retrieval: %.3f sec (%d results)",
            bm25_time,
            len(bm25_results),
        )

        start = time.perf_counter()

        merged_results = self._merge_results(
            dense_results,
            bm25_results,
        )

        merge_time = time.perf_counter() - start

        logger.debug(
            "Hybrid merge: %.3f sec (%d results)",
            merge_time,
            len(merged_results),
        )

        return merged_results[:top_k]
