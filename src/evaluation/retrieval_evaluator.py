from src.core.models import BenchmarkCase, BenchmarkResult, SearchResult
from src.evaluation.retrieval_metrics import RetrievalMetric


class RetrievalEvaluator:

    def __init__(
        self,
        metrics: list[RetrievalMetric],
    ) -> None:

        self.metrics = metrics

    def evaluate(
        self,
        case: BenchmarkCase,
        retrieved: list[SearchResult],
    ) -> BenchmarkResult:

        metric_scores = {}

        for metric in self.metrics:

            metric_scores[metric.name] = metric.compute(
                case,
                retrieved,
            )

        passed = metric_scores["Recall@K"] == 1.0

        return BenchmarkResult(
            case_id=case.id,
            question=case.question,
            retrieved_chunks=retrieved,
            metric_scores=metric_scores,
            passed=passed,
        )
