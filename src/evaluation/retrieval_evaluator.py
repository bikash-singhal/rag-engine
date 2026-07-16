from src.core.models import RetrievalEvaluation, SearchResult


class RetrievalEvaluator:

    def evaluate(
        self,
        results: list[SearchResult],
        expected_chunk_ids: set[int],
    ) -> RetrievalEvaluation:

        if not expected_chunk_ids:
            raise ValueError("The provided Benchmark Dataset cannot be empty.")

        retrieved_chunk_ids = {result.chunk.chunk_index for result in results}

        hits = retrieved_chunk_ids & expected_chunk_ids

        precision = self._precision(
            hits,
            retrieved_chunk_ids,
        )

        recall = self._recall(
            hits,
            expected_chunk_ids,
        )

        f1_score = self._f1_score(
            precision,
            recall,
        )

        mrr = self._mrr(results, expected_chunk_ids)

        return RetrievalEvaluation(
            hit_rate=self._hit_rate(hits),
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            mrr=mrr,
        )

    def _hit_rate(
        self,
        hits: set[int],
    ) -> float:

        return 1.0 if hits else 0.0

    def _precision(
        self,
        hits: set[int],
        retrieved_chunk_ids: set[int],
    ) -> float:

        if not retrieved_chunk_ids:
            return 0.0

        return len(hits) / len(retrieved_chunk_ids)

    def _recall(
        self,
        hits: set[int],
        expected_chunk_ids: set[int],
    ) -> float:

        return len(hits) / len(expected_chunk_ids)

    def _f1_score(
        self,
        precision: float,
        recall: float,
    ) -> float:

        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)

    def _mrr(
        self,
        results: list[SearchResult],
        expected_chunk_ids: set[int],
    ) -> float:

        for position, result in enumerate(results, start=1):

            if result.chunk.chunk_index in expected_chunk_ids:

                return 1 / position

        return 0
