from dotenv import load_dotenv

from src.app.rag_engine import RAGEngine
from utilities.benchmark_generator import BenchmarkGenerator

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


def main():

    load_dotenv()

    engine = RAGEngine()

    engine.load_or_ingest(
        document_directory="data/raw",
        index_directory="data/indexes/default",
    )

    builder = BenchmarkGenerator(
        retriever=engine.retriever,
    )

    for chunk in engine.vector_store.get_chunks():
        if "The original Amazon SageMaker has been renamed SageMaker AI" in chunk.text:
            print(chunk.chunk_index)
            print(chunk.page_number)
            print(chunk.text)

    benchmark = builder.build(QUESTIONS)

    print("\n" + "=" * 80)
    print("Copy this into benchmark.py")
    print("=" * 80 + "\n")

    print("benchmark = [")

    for question, chunk_ids in benchmark:

        print("    (")
        print(f'        "{question}",')
        print(f"        {chunk_ids},")
        print("    ),")

    print("]")


if __name__ == "__main__":
    main()
