from src.config.settings import RETRIEVAL_TOP_K
from src.core.models import BenchmarkDataset, BenchmarkSummary
from src.evaluation.retrieval_evaluator import RetrievalEvaluator


class BenchmarkRunner:

    def __init__(
        self,
        retriever,
        evaluator: RetrievalEvaluator,
    ) -> None:

        self.retriever = retriever
        self.evaluator = evaluator

    def run(
        self,
        dataset: BenchmarkDataset,
        experiment_name: str,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> BenchmarkSummary:

        results = []

        for case in dataset.cases:

            retrieved = self.retriever.retrieve(
                query=case.question,
                top_k=top_k,
            )

            result = self.evaluator.evaluate(
                case,
                retrieved,
            )

            results.append(result)

        if not results:
            raise ValueError("Benchmark dataset contains no test cases.")

        passed = sum(r.passed for r in results)

        metric_scores = {}

        for metric in self.evaluator.metrics:

            metric_scores[metric.name] = sum(
                result.metric_scores[metric.name] for result in results
            ) / len(results)

        return BenchmarkSummary(
            experiment_name=experiment_name,
            benchmark_name=dataset.name,
            total_cases=len(results),
            metric_scores=metric_scores,
            passed_cases=passed,
            failed_cases=len(results) - passed,
            results=results,
        )
