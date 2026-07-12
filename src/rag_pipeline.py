from pathlib import Path

from src.models import Chunk, SearchResult
from src.pdf_reader import read_pdf
from src.chunker import Chunker
from src.embedder import Embedder
from src.faiss_store import FAISSVectorStore
from src.prompt_builder import PromptBuilder
from src.llm import LLM


class RAGPipeline:
    """
    Coordinates the complete RAG workflow.
    """

    def __init__(
        self,
        chunker: Chunker,
        embedder: Embedder,
        vector_store: FAISSVectorStore,
        prompt_builder: PromptBuilder,
        llm: LLM,
    ) -> None:

        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.prompt_builder = prompt_builder
        self.llm = llm

    def index(
        self,
        pdf_path: str | Path,
    ) -> None:
        """
        Reads, chunks, embeds and indexes a PDF.
        """

        document = read_pdf(pdf_path)

        chunks = self.chunker.chunk(document)

        embedded_chunks = self.embedder.embed(chunks)

        self.vector_store.add(embedded_chunks)

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Retrieves the most relevant chunks.
        """

        query_chunk = Chunk(
            chunk_index=-1,
            source="query",
            text=question,
        )

        query_embedding = (
            self.embedder
            .embed([query_chunk])[0]
            .embedding
        )

        return self.vector_store.search(
            query_embedding,
            top_k,
        )

    def ask(
        self,
        question: str,
        top_k: int = 5,
    ) -> str:
        """
        Executes the complete RAG pipeline.
        """

        results = self.retrieve(
            question,
            top_k,
        )

        prompt = self.prompt_builder.build(
            question,
            results,
        )

        return self.llm.generate(prompt)