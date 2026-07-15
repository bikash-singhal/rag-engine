import os
from pathlib import Path

from src.config.settings import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL
from src.core.models import SearchResult
from src.embeddings.embedder import Embedder
from src.ingestion.chunker import Chunker
from src.ingestion.indexer import DocumentIndexer
from src.ingestion.preprocessor import Preprocessor
from src.ingestion.reader import reader
from src.llm.bedrock_provider import BedrockProvider
from src.rag.rag_pipeline import RAGPipeline
from src.vectorstores.faiss_store import FAISSVectorStore


class RAGEngine:
    """
    High-level interface for the RAG system.
    """

    def __init__(
        self,
    ) -> None:
        self.embedder = Embedder(
            model_name=EMBEDDING_MODEL,
        )

        self.vector_store = FAISSVectorStore(
            embedding_dim=self.embedder.embedding_dimension,
        )

        self.indexer = DocumentIndexer(
            reader=reader,
            preprocessor=Preprocessor(),
            chunker=Chunker(
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP,
            ),
            embedder=self.embedder,
            vector_store=self.vector_store,
        )

        model_id = os.getenv("BEDROCK_MODEL")

        if model_id is None:
            raise RuntimeError("BEDROCK_MODEL environment variable is not set.")

        self.llm_provider = BedrockProvider(model_id)

        self.rag = RAGPipeline(
            embedder=self.embedder,
            vector_store=self.vector_store,
            llm=self.llm_provider,
        )

    def ask(
        self,
        question: str,
    ) -> str:

        return self.rag.ask(question)

    def ingest(
        self,
        pdf_file: str | Path,
    ) -> None:

        self.indexer.index(pdf_file)

    def ingest_directory(
        self,
        directory: str | Path,
    ) -> None:

        self.indexer.index_directory(directory)

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
    ) -> list[SearchResult]:

        return self.rag.retrieve(
            question,
            top_k,
        )

    def save_index(
        self,
        index_directory: str | Path,
    ) -> None:

        self.vector_store.save(index_directory)

    def load_index(
        self,
        index_directory: str | Path,
    ) -> None:

        self.vector_store = FAISSVectorStore.load(index_directory)

        self.rag.vector_store = self.vector_store
        self.indexer.vector_store = self.vector_store

    def load_or_ingest(
        self,
        document_directory: str | Path,
        index_directory: str | Path,
    ) -> None:
        """
        Loads an existing vector index if available.
        Otherwise ingests the documents and saves the index.
        """

        index_directory = Path(index_directory)

        faiss_file = index_directory / "faiss.index"
        metadata_file = index_directory / "metadata.pkl"

        if faiss_file.exists() and metadata_file.exists():

            print("\nLoading existing vector index...\n")

            self.load_index(index_directory)

            print(f"Loaded {self.vector_store.index.ntotal} vectors.")

            return

        print("\nNo existing index found.\n")

        print("Building vector index...\n")

        self.ingest_directory(document_directory)

        self.save_index(index_directory)

        print("\nVector index saved.\n")
