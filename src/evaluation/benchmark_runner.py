from src.core.models import BenchmarkEvaluation, RetrievalEvaluation
from src.evaluation.retrieval_evaluator import RetrievalEvaluator
from src.retrieval.retriever import Retriever


class BenchmarkRunner:
    """
    Runs a retrieval benchmark over a collection of questions.
    """

    def __init__(
        self,
        retriever: Retriever,
        evaluator: RetrievalEvaluator,
    ) -> None:

        self.retriever = retriever
        self.evaluator = evaluator

    def run(
        self,
        benchmark: list[tuple[str, set[int]]],
        top_k: int = 5,
    ) -> BenchmarkEvaluation:
        """
        Executes the benchmark and returns aggregated metrics.
        """

        if not benchmark:
            raise ValueError("Benchmark cannot be empty.")

        evaluations: list[RetrievalEvaluation] = []

        for question, expected_chunk_ids in benchmark:

            results = self.retriever.retrieve(
                query=question,
                top_k=top_k,
            )

            evaluation = self.evaluator.evaluate(
                results=results,
                expected_chunk_ids=expected_chunk_ids,
            )

            evaluations.append(evaluation)

        average_hit_rate = self._average(
            [evaluation.hit_rate for evaluation in evaluations]
        )
        average_precision = self._average(
            [evaluation.precision for evaluation in evaluations]
        )
        average_recall = self._average(
            [evaluation.recall for evaluation in evaluations]
        )
        average_f1_score = self._average(
            [evaluation.f1_score for evaluation in evaluations]
        )
        average_mrr = self._average([evaluation.mrr for evaluation in evaluations])

        return BenchmarkEvaluation(
            average_hit_rate=average_hit_rate,
            average_precision=average_precision,
            average_recall=average_recall,
            average_f1_score=average_f1_score,
            average_mrr=average_mrr,
        )

    def _average(
        self,
        values: list[float],
    ) -> float:

        if not values:
            return 0.0

        return sum(values) / len(values)
