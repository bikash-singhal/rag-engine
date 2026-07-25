from src.config.settings import RETRIEVAL_TOP_K
from src.core.models import BenchmarkDataset, BenchmarkSummary
from src.evaluation.generation_evaluator import GenerationEvaluator
from src.evaluation.retrieval_evaluator import RetrievalEvaluator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BenchmarkRunner:

    def __init__(
        self,
        chat_engine,
        retrieval_evaluator: RetrievalEvaluator,
        generation_evaluator: GenerationEvaluator | None = None,
    ):
        self.chat_engine = chat_engine
        self.retrieval_evaluator = retrieval_evaluator
        self.generation_evaluator = generation_evaluator

    def run(
        self,
        dataset: BenchmarkDataset,
        experiment_name: str,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> BenchmarkSummary:

        results = []

        for index, case in enumerate(dataset.cases, start=1):

            chat_result = self.chat_engine.evaluate(case.question)
            retrieved = chat_result.retrieved_chunks

            result = self.retrieval_evaluator.evaluate(
                case,
                retrieved,
            )

            if self.generation_evaluator:
                try:
                    generation_scores = self.generation_evaluator.evaluate(
                        question=case.question,
                        contexts=[item.chunk.text for item in retrieved],
                        answer=chat_result.answer,
                        expected_answer=getattr(
                            case,
                            "expected_answer",
                            None,
                        ),
                    )

                except Exception as exc:
                    logger.error(
                        "Generation evaluation failed. "
                        "case=%d question=%r "
                        "error_type=%s error=%s",
                        index,
                        case.question,
                        type(exc).__name__,
                        exc,
                        exc_info=True,
                    )
                    generation_scores = {}

                result.metric_scores.update(generation_scores)

            self._print_case_result(
                index=index,
                total_cases=len(dataset.cases),
                question=case.question,
                metric_scores=result.metric_scores,
            )

            results.append(result)

        if not results:
            raise ValueError("Benchmark dataset contains no test cases.")

        passed = sum(result.passed for result in results)

        metric_scores = self._calculate_average_metrics(results)

        return BenchmarkSummary(
            experiment_name=experiment_name,
            benchmark_name=dataset.name,
            total_cases=len(results),
            metric_scores=metric_scores,
            passed_cases=passed,
            failed_cases=len(results) - passed,
            results=results,
        )

    @staticmethod
    def _print_case_result(
        index: int,
        total_cases: int,
        question: str,
        metric_scores: dict[str, float],
    ) -> None:

        recall = metric_scores.get("Recall@K")
        mrr = metric_scores.get("MRR")
        faithfulness = metric_scores.get("faithfulness")

        recall_display = f"{recall:.3f}" if recall is not None else "N/A"
        mrr_display = f"{mrr:.3f}" if mrr is not None else "N/A"
        faithfulness_display = (
            f"{faithfulness:.3f}" if faithfulness is not None else "N/A"
        )

        print("-" * 80)
        print(f"Case {index}/{total_cases}")
        print(f"Question : {question}")
        print()
        print(
            f"Recall={recall_display} | "
            f"MRR={mrr_display} | "
            f"Faith={faithfulness_display}"
        )

        notes = []

        if mrr is not None and mrr < 1.0:
            notes.append("Retrieval")

        if faithfulness is not None and faithfulness < 1.0:
            notes.append("Faithfulness")

        if faithfulness is None:
            notes.append("Faithfulness unavailable")

        if notes:
            print(f"Observations: {', '.join(notes)}")

    @staticmethod
    def _calculate_average_metrics(
        results,
    ) -> dict[str, float]:

        all_metric_names = {
            metric_name for result in results for metric_name in result.metric_scores
        }

        average_scores = {}

        for metric_name in all_metric_names:
            available_scores = [
                result.metric_scores[metric_name]
                for result in results
                if metric_name in result.metric_scores
            ]

            if not available_scores:
                continue

            average_scores[metric_name] = sum(available_scores) / len(available_scores)

            if len(available_scores) < len(results):
                logger.warning(
                    "Metric was unavailable for some cases. "
                    "metric=%s evaluated_cases=%d total_cases=%d",
                    metric_name,
                    len(available_scores),
                    len(results),
                )

        return average_scores
