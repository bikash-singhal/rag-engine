from src.core.models import SearchResult
from src.embeddings.embedder import Embedder
from src.llm.base import LLMProvider
from src.rag.prompt_builder import PromptBuilder
from src.vectorstores.faiss_store import FAISSVectorStore


class RAGPipeline:
    """
    Coordinates retrieval and answer generation.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: FAISSVectorStore,
        llm: LLMProvider,
    ) -> None:

        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Retrieves the most relevant chunks for a user question.
        """

        query_embedding = self.embedder.embed_query(
            question
        )

        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

    def ask(
        self,
        question: str,
        top_k: int = 5,
    ) -> str:
        """
        Executes the complete Retrieval-Augmented Generation workflow.
        """

        results = self.retrieve(
            question=question,
            top_k=top_k,
        )

        prompt = PromptBuilder.build(
            question=question,
            results=results,
        )

        return self.llm.generate(prompt)