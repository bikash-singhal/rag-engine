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
