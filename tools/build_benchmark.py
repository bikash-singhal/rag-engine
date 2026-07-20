from pathlib import Path

from dotenv import load_dotenv

from src.app.rag_engine import RAGEngine
from utilities.benchmark_builder import BenchmarkBuilder

QUESTIONS = [
    "What is Amazon SageMaker?",
    "What are the two primary components of Amazon SageMaker?",
    "What is SageMaker Unified Studio?",
    "What is SageMaker AI?",
    "What is Data Processing?",
    "What is Data and AI Governance?",
    "What is the Lakehouse architecture?",
    "What frameworks are supported for data processing?",
    "What data sources can SageMaker connect to?",
    "What are the prerequisites for Amazon SageMaker?",
]


def main() -> None:

    load_dotenv()

    engine = RAGEngine()

    engine.load_or_ingest(
        document_directory="data/raw",
        index_directory="data/indexes/default",
    )

    builder = BenchmarkBuilder(
        retriever=engine.retriever,
    )

    dataset = builder.build(
        questions=QUESTIONS,
        benchmark_name="Amazon SageMaker Benchmark",
        description="Retrieval benchmark for Amazon SageMaker documentation.",
        version="1.0",
        top_k=10,
    )

    output_path = Path(
        "data/benchmark/sagemaker.json",
    )

    builder.save(
        dataset=dataset,
        output_file=output_path,
    )

    print(f"\nBenchmark saved to:\n{output_path.resolve()}")


if __name__ == "__main__":
    main()
