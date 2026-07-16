from src.llm.base import LLMProvider
from src.rag.prompt_builder import PromptBuilder
from src.retrieval.retriever import Retriever


class RAGPipeline:
    """
    Coordinates retrieval and answer generation.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMProvider,
    ) -> None:

        self.retriever = retriever
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
            top_k,
        )

        prompt = PromptBuilder.build(
            question=question,
            results=results,
        )

        return self.llm.generate(prompt)
