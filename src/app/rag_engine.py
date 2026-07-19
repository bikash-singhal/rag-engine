import os
from collections.abc import Iterator
from pathlib import Path

from src.chat.chat_engine import ChatEngine
from src.chat.in_memory import InMemoryMemory
from src.config.settings import (
    BM25_WEIGHT,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DENSE_WEIGHT,
    EMBEDDING_MODEL,
    RETRIEVAL_TOP_K,
)
from src.core.models import SearchResult
from src.embeddings.embedder import Embedder
from src.evaluation.retrieval_analyzer import RetrievalAnalyzer
from src.ingestion.chunker import Chunker
from src.ingestion.indexer import DocumentIndexer
from src.ingestion.preprocessor import Preprocessor
from src.ingestion.reader import reader
from src.llm.bedrock_provider import BedrockProvider
from src.prompt.prompt_builder import PromptBuilder
from src.query.llm_multi_query_generator import LLMMultiQueryGenerator
from src.query.llm_query_rewriter import LLMQueryRewriter
from src.query.rewrite_prompt import RewritePromptBuilder
from src.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranking_retriever import RerankingRetriever
from src.utils.logger import get_logger
from src.vectorstores.faiss_store import FAISSVectorStore

logger = get_logger(__name__)


class RAGEngine:
    """
    High-level interface for the RAG system.
    """

    def __init__(
        self,
    ) -> None:
        logger.info("Initializing RAG Engine...")

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

        logger.info(
            "Using LLM model: %s",
            model_id,
        )

        self.retrieval_analyzer = RetrievalAnalyzer()

        logger.info("RAG Engine initialized.")

    def _build_chat_engine(self) -> None:
        logger.info("Building chat engine...")

        chunks = self.vector_store.get_chunks()

        logger.info(
            "Loaded %d chunks from vector store.",
            len(chunks),
        )

        dense_retriever = DenseRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
        )

        bm25_retriever = BM25Retriever(chunks)

        self.hybrid_retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            bm25_retriever=bm25_retriever,
            dense_weight=DENSE_WEIGHT,
            bm25_weight=BM25_WEIGHT,
        )

        reranker = CrossEncoderReranker()

        self.reranking_retriever = RerankingRetriever(
            retriever=self.hybrid_retriever,
            reranker=reranker,
        )

        # Default retriever used by the application
        self.retriever = self.hybrid_retriever

        memory = InMemoryMemory()

        prompt_builder = PromptBuilder()

        rewrite_prompt_builder = RewritePromptBuilder()

        query_rewriter = LLMQueryRewriter(
            llm=self.llm_provider,
            prompt_builder=rewrite_prompt_builder,
        )

        multi_query_generator = LLMMultiQueryGenerator(self.llm_provider)

        self.chat_engine = ChatEngine(
            memory=memory,
            query_rewriter=query_rewriter,
            retriever=self.retriever,
            reranker=reranker,
            multi_query_generator=multi_query_generator,
            prompt_builder=prompt_builder,
            llm=self.llm_provider,
        )

    def ask(
        self,
        question: str,
    ) -> Iterator[str]:

        return self.chat_engine.ask(question)

    def ingest(
        self,
        pdf_file: str | Path,
    ) -> None:

        self.indexer.index(pdf_file)

    def ingest_directory(
        self,
        directory: str | Path,
    ) -> None:

        logger.info(
            "Indexing directory: %s",
            directory,
        )

        self.indexer.index_directory(directory)

        logger.info("Directory indexed.")

    def retrieve(
        self,
        question: str,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> list[SearchResult]:

        return self.retriever.retrieve(
            question,
            top_k,
        )

    def save_index(
        self,
        index_directory: str | Path,
    ) -> None:
        logger.info(
            "Saving vector index to %s",
            index_directory,
        )

        self.vector_store.save(index_directory)

    def load_index(
        self,
        index_directory: str | Path,
    ) -> None:

        logger.info(
            "Loading vector index from %s",
            index_directory,
        )

        self.vector_store = FAISSVectorStore.load(index_directory)

        self.indexer.vector_store = self.vector_store
        self._build_chat_engine()
        logger.info("Chat engine rebuilt.")

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

            logger.info("Loading existing vector index...")

            self.load_index(index_directory)

            logger.info("Loaded vector count: %d", self.vector_store.index.ntotal)

            return

        logger.info("No existing index found.")

        logger.info("Building vector index...")

        self.ingest_directory(document_directory)
        self._build_chat_engine()
        self.save_index(index_directory)

        logger.info("Vector index saved.")

    def evaluate(
        self,
        question: str,
    ):
        """
        Retrieves relevant chunks and returns a retrieval report.
        """
        logger.info("Running retrieval evaluation...")
        results = self.retrieve(question)

        return self.retrieval_analyzer.analyze(
            question,
            results,
        )
