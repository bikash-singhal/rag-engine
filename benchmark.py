from dotenv import load_dotenv

from src.app.rag_engine import RAGEngine
from src.evaluation.benchmark_runner import BenchmarkRunner
from src.evaluation.experiment_runner import ExperimentRunner
from src.evaluation.report_formatter import ReportFormatter
from src.evaluation.retrieval_evaluator import RetrievalEvaluator


def main():

    load_dotenv()

    engine = RAGEngine()

    engine.load_or_ingest(
        document_directory="data/raw",
        index_directory="data/indexes/default",
    )

    benchmark = [
        (
            "What is SageMaker?",
            {2},
        ),
        (
            "What is Bedrock?",
            {56},
        ),
    ]

    evaluator = RetrievalEvaluator()

    runner = BenchmarkRunner(
        retriever=engine.retriever,
        evaluator=evaluator,
    )

    benchmark_result = runner.run(
        benchmark=benchmark,
    )

    ReportFormatter.print_benchmark_report(
        benchmark_result,
    )

    print()

    report = ExperimentRunner(
        evaluator=evaluator,
    ).run(
        retrievers={
            "Hybrid": engine.retriever,
        },
        benchmark=benchmark,
    )

    ReportFormatter.print_experiment_report(report)


if __name__ == "__main__":
    main()
