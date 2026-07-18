from dotenv import load_dotenv

from src.app.rag_engine import RAGEngine
from src.config.settings import BM25_WEIGHT, DENSE_WEIGHT
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

    # TODO: Expand benchmark dataset to 30–50 questions after retrieval pipeline is finalized.

    benchmark = [
        (
            "What is Amazon SageMaker?",
            {6},
        ),
        (
            "What are the two primary components of Amazon SageMaker?",
            {6},
        ),
        (
            "What is SageMaker Unified Studio?",
            {6},
        ),
        (
            "What is SageMaker AI?",
            {6},
        ),
        (
            "What is Data Processing?",
            {11},
        ),
        (
            "What is Data and AI Governance?",
            {6},
        ),
        (
            "What is the Lakehouse architecture?",
            {9},
        ),
        (
            "What frameworks are supported for data processing?",
            {10},
        ),
        (
            "What data sources can SageMaker connect to?",
            {11},
        ),
        (
            "What are the prerequisites for Amazon SageMaker?",
            {15},
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
            "Hybrid": engine.hybrid_retriever,
            "Hybrid + CrossEncoder": engine.reranking_retriever,
        },
        benchmark=benchmark,
    )

    ReportFormatter.print_experiment_report(report)


if __name__ == "__main__":
    main()
