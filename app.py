from argparse import ArgumentParser

from dotenv import load_dotenv

from src.config.settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_PROVIDER,
    TOP_K,
)

from src.llm.factory import get_provider
from src.ingestion.chunker import Chunker
from src.embeddings.embedder import Embedder
from src.vectorstores.faiss_store import FAISSVectorStore
from src.rag.prompt_builder import PromptBuilder
from src.rag.rag_pipeline import RAGPipeline


def parse_args():
    parser = ArgumentParser(
        description="Retrieval-Augmented Generation (RAG) Engine"
    )

    parser.add_argument(
        "--pdf",
        required=True,
        help="Path to the PDF document.",
    )

    parser.add_argument(
        "--question",
        required=True,
        help="Question to ask.",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    load_dotenv()

    chunker = Chunker(
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )

    embedder = Embedder(
        model_name=EMBEDDING_MODEL,
    )

    vector_store = FAISSVectorStore(
        embedding_dim=embedder.embedding_dimension,
    )

    prompt_builder = PromptBuilder()

    llm = get_provider(
        provider_name=LLM_PROVIDER,
        model=LLM_MODEL,
    )

    pipeline = RAGPipeline(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        prompt_builder=prompt_builder,
        llm=llm,
    )

    print("Indexing document...")

    pipeline.index(args.pdf)

    print("Document indexed successfully.\n")

    answer = pipeline.ask(
        args.question,
        top_k=TOP_K,
    )

    print("=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(args.question)

    print("\n")

    print("=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(answer)


if __name__ == "__main__":
    main()