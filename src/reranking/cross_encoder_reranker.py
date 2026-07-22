from sentence_transformers import CrossEncoder

from src.config.settings import MAX_RERANK_CANDIDATES, RERANKER_MODEL
from src.core.models import SearchResult
from src.reranking.reranker import Reranker
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker(Reranker):

    def __init__(
        self,
        max_candidates: int = MAX_RERANK_CANDIDATES,
    ) -> None:

        if not RERANKER_MODEL:
            raise RuntimeError("RERANKER_MODEL is not configured.")

        self.model = CrossEncoder(RERANKER_MODEL)
        self.max_candidates = max_candidates

        logger.info(
            "CrossEncoder device: %s",
            self.model.device,
        )

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = MAX_RERANK_CANDIDATES,
    ) -> list[SearchResult]:

        if not results:
            return []

        logger.debug(
            "Received %d candidates: ",
            len(results),
        )

        unique_results = self._deduplicate_results(results)

        logger.debug(
            "Deduplicated to  %d unique candidates: ",
            len(unique_results),
        )

        logger.debug("Unique candidates:")

        for result in unique_results:
            logger.debug(
                "%s | chunk=%d | %.4f",
                result.chunk.source,
                result.chunk.chunk_index,
                result.score,
            )

        pruned_results = self._prune(
            max_canidates=self.max_candidates,
            results=unique_results,
        )

        logger.debug("Candidates after pruning:")

        for result in pruned_results:
            logger.debug(
                "%s | chunk=%d | %.4f",
                result.chunk.source,
                result.chunk.chunk_index,
                result.score,
            )

        pairs = [(query, result.chunk.text) for result in pruned_results]

        # scores = self.model.predict(pairs)

        from time import perf_counter

        logger.info(
            "Running CrossEncoder on %d pairs",
            len(pairs),
        )

        start = perf_counter()

        scores = self.model.predict(pairs)

        logger.info(
            "CrossEncoder.predict(): %.2f ms",
            (perf_counter() - start) * 1000,
        )

        reranked_results = [
            SearchResult(
                chunk=result.chunk,
                score=float(score),
            )
            for result, score in zip(pruned_results, scores)
        ]

        reranked_results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        logger.debug("Top after CrossEncoder:")

        for result in reranked_results:
            logger.debug(
                "%s | chunk=%d | %.4f",
                result.chunk.source,
                result.chunk.chunk_index,
                result.score,
            )

        return reranked_results[:top_k]

    def _deduplicate_results(
        self,
        results: list[SearchResult],
    ) -> list[SearchResult]:

        seen = set()

        unique_results = []

        for result in results:

            if result.chunk.id in seen:
                continue

            seen.add(result.chunk.id)

            unique_results.append(result)

        return unique_results

    def _prune(
        self,
        max_canidates: int,
        results: list[SearchResult],
    ) -> list[SearchResult]:

        return results[:max_canidates]
