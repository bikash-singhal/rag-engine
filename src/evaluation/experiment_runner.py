from src.core.models import ExperimentReport, ExperimentResult
from src.evaluation.benchmark_runner import BenchmarkRunner
from src.evaluation.retrieval_evaluator import RetrievalEvaluator
from src.retrieval.retriever import Retriever


class ExperimentRunner:
    """
    Runs benchmark experiments across multiple retrievers.
    """

    def __init__(
        self,
        evaluator: RetrievalEvaluator,
    ) -> None:

        self.evaluator = evaluator

    def run(
        self,
        retrievers: dict[str, Retriever],
        benchmark: list[tuple[str, set[int]]],
        top_k: int = 5,
    ) -> ExperimentReport:

        if not retrievers:
            raise ValueError("Retrievers cannot be empty.")

        if not benchmark:
            raise ValueError("Benchmark cannot be empty.")

        experiments: list[ExperimentResult] = []

        for name, retriever in retrievers.items():

            benchmark_runner = BenchmarkRunner(
                retriever=retriever,
                evaluator=self.evaluator,
            )

            benchmark_result = benchmark_runner.run(
                benchmark=benchmark,
                top_k=top_k,
            )

            experiments.append(
                ExperimentResult(
                    name=name,
                    benchmark=benchmark_result,
                )
            )

        return ExperimentReport(
            experiments=experiments,
        )
