from __future__ import annotations

from src.core.models import BenchmarkDataset, BenchmarkSummary, Experiment
from src.evaluation.benchmark_runner import BenchmarkRunner
from src.evaluation.retrieval_evaluator import RetrievalEvaluator


class ExperimentRunner:
    """
    Runs multiple retrieval experiments against the same benchmark dataset.
    """

    def __init__(
        self,
        evaluator: RetrievalEvaluator,
        top_k: int,
    ) -> None:

        self.evaluator = evaluator
        self.top_k = top_k

    def run(
        self,
        dataset: BenchmarkDataset,
        experiments: list[Experiment],
    ) -> list[BenchmarkSummary]:

        summaries: list[BenchmarkSummary] = []

        for experiment in experiments:

            runner = BenchmarkRunner(
                retriever=experiment.retriever,
                evaluator=self.evaluator,
            )

            summary = runner.run(
                dataset=dataset,
                experiment_name=experiment.name,
                top_k=self.top_k,
            )

            summaries.append(summary)

        return summaries
