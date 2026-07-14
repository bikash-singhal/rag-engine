from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.config.settings import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    LLM_MODEL,
)

from src.ingestion.chunker import Chunker
from src.embeddings.embedder import Embedder
from src.vectorstores.faiss_store import FAISSVectorStore
from src.llm import LLM
from src.pdf_reader import read_pdf
from src.rag.prompt_builder import PromptBuilder
from src.rag.rag_pipeline import RAGPipeline


def main() -> None:
    load_dotenv()

    print("=" * 60)
    print("RAG SANITY CHECK")
    print("=" * 60)

    pdf_path = input("\nEnter PDF path: ").strip()

    if not Path(pdf_path).exists():
        raise FileNotFoundError(pdf_path)

    # ---------------------------------------------------
    # PDF Reader
    # ---------------------------------------------------
    print("\n[1] Testing PDF Reader...")

    document = read_pdf(pdf_path)

    print(f"Source : {document.source}")
    print(f"Characters : {len(document.text)}")

    # ---------------------------------------------------
    # Chunker
    # ---------------------------------------------------
    print("\n[2] Testing Chunker...")

    chunker = Chunker(
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )

    chunks = chunker.chunk(document)

    print(f"Chunks created : {len(chunks)}")
    print(f"First chunk length : {len(chunks[0].text.split())}")

    # ---------------------------------------------------
    # Embedder
    # ---------------------------------------------------
    print("\n[3] Testing Embedder...")

    embedder = Embedder(
        model_name=EMBEDDING_MODEL,
    )

    embedded_chunks = embedder.embed(chunks)

    print(f"Embedding dimension : {embedder.embedding_dimension}")
    print(f"Embedded chunks : {len(embedded_chunks)}")

    # ---------------------------------------------------
    # Vector Store
    # ---------------------------------------------------
    print("\n[4] Testing FAISS...")

    vector_store = FAISSVectorStore(
        embedding_dim=embedder.embedding_dimension,
    )

    vector_store.add(embedded_chunks)

    print(f"Indexed vectors : {vector_store.index.ntotal}")

    # ---------------------------------------------------
    # Search
    # ---------------------------------------------------
    print("\n[5] Testing Retrieval...")

    query = "What is this document about?"

    query_embedding = embedder.model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=3,
    )

    print(f"Retrieved {len(results)} chunks")

    # ---------------------------------------------------
    # Prompt Builder
    # ---------------------------------------------------
    print("\n[6] Testing Prompt Builder...")

    prompt_builder = PromptBuilder()

    prompt = prompt_builder.build(
        question=query,
        results=results,
    )

    print(f"Prompt length : {len(prompt)} characters")

    # ---------------------------------------------------
    # LLM
    # ---------------------------------------------------
    print("\n[7] Testing OpenAI...")

    client = OpenAI()

    llm = LLM(
        client=client,
        model=LLM_MODEL,
    )

    answer = llm.generate(prompt)

    print("\nLLM Response:")
    print("-" * 60)
    print(answer[:500])
    print("-" * 60)

    # ---------------------------------------------------
    # Full Pipeline
    # ---------------------------------------------------
    print("\n[8] Testing Complete Pipeline...")

    pipeline = RAGPipeline(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        prompt_builder=prompt_builder,
        llm=llm,
    )

    final_answer = pipeline.ask(
        "Summarize this document.",
        top_k=3,
    )

    print("\nPipeline Response:")
    print("-" * 60)
    print(final_answer[:500])
    print("-" * 60)

    print("\n✅ All sanity checks passed!")


if __name__ == "__main__":
    main()