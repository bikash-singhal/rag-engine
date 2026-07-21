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
from src.core.models import RetrievalReport, SearchResult
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

    def __init__(self) -> None:

        logger.info("Initializing RAG Engine...")

        # ------------------------------------------------------------------
        # Infrastructure
        # ------------------------------------------------------------------

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

        # ------------------------------------------------------------------
        # LLM
        # ------------------------------------------------------------------

        model_id = os.getenv("BEDROCK_MODEL")

        if model_id is None:
            raise RuntimeError("BEDROCK_MODEL environment variable is not set.")

        self.llm_provider = BedrockProvider(model_id)

        logger.info(
            "Using LLM model: %s",
            model_id,
        )

        # ------------------------------------------------------------------
        # Evaluation
        # ------------------------------------------------------------------

        self.retrieval_analyzer = RetrievalAnalyzer()

        logger.info("RAG Engine initialized.")

    def _build_chat_engine(self) -> None:

        logger.info("Building chat engine...")

        memory = InMemoryMemory()

        prompt_builder = PromptBuilder()

        rewrite_prompt_builder = RewritePromptBuilder()

        query_rewriter = LLMQueryRewriter(
            llm=self.llm_provider,
            prompt_builder=rewrite_prompt_builder,
        )

        multi_query_generator = LLMMultiQueryGenerator(
            self.llm_provider,
        )

        self.chat_engine = ChatEngine(
            memory=memory,
            query_rewriter=query_rewriter,
            retriever=self.retriever,
            reranker=self.reranker,
            multi_query_generator=multi_query_generator,
            prompt_builder=prompt_builder,
            llm=self.llm_provider,
        )

        logger.info("Chat engine built.")

    def _build_retriever(self) -> None:

        logger.info("Building retrieval pipeline...")

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

        self.reranker = CrossEncoderReranker()

        self.reranking_retriever = RerankingRetriever(
            retriever=self.hybrid_retriever,
            reranker=self.reranker,
        )

        # Default retriever
        self.retriever = self.hybrid_retriever

    def ask(
        self,
        question: str,
    ) -> Iterator[str]:

        return self.chat_engine.ask(question)

    def ingest(
        self,
        pdf_file: str | Path,
    ) -> None:

        logger.info(
            "Indexing document: %s",
            pdf_file,
        )

        self.indexer.index(pdf_file)

        self._reload_runtime_components()

        logger.info("Document indexed and retriever refreshed.")

    def ingest_directory(
        self,
        directory: str | Path,
    ) -> None:

        logger.info(
            "Indexing directory: %s",
            directory,
        )

        self.indexer.index_directory(directory)

        self._reload_runtime_components()

        logger.info("Directory indexed and refreshed retriever.")

    def _reload_runtime_components(self) -> None:

        self._build_retriever()
        self._build_chat_engine()

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

        logger.info("Vector index saved.")

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

        logger.info(
            "Loaded %d vectors.",
            self.vector_store.index.ntotal,
        )

        self._reload_runtime_components()

    def _create_index(
        self,
        document_directory,
        index_directory,
    ) -> None:

        logger.info("Building new vector index...")

        self.ingest_directory(document_directory)
        self._reload_runtime_components()
        self.save_index(index_directory)

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

            self.load_index(index_directory)

            return

        logger.info("No existing index found.")

        self._create_index(
            document_directory=document_directory,
            index_directory=index_directory,
        )

    def evaluate(
        self,
        question: str,
    ) -> RetrievalReport:
        """
        Runs retrieval evaluation for a question.
        """

        logger.info(
            "Running retrieval evaluation for question: %s",
            question,
        )

        results = self.retrieve(question)

        return self.retrieval_analyzer.analyze(
            question,
            results,
        )
