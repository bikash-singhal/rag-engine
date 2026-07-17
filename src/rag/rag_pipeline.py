from src.llm.base import LLMProvider
from src.rag.prompt_builder import PromptBuilder
from src.reranking.reranker import Reranker
from src.retrieval.retriever import Retriever


class RAGPipeline:
    """
    Coordinates retrieval and answer generation.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMProvider,
        reranker: Reranker | None = None,
    ) -> None:

        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm

    def ask(
        self,
        question: str,
        top_k: int = 5,
    ) -> str:
        """
        Executes the complete Retrieval-Augmented Generation workflow.
        """

        results = self.retriever.retrieve(
            question,
            top_k=20,
        )

        if self.reranker is not None:
            results = self.reranker.rerank(
                query=question,
                results=results,
            )

        prompt = PromptBuilder.build(
            question=question,
            results=results,
        )

        return self.llm.generate(prompt)
