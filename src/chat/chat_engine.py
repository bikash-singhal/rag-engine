from collections.abc import Iterator

from src.chat.memory import Memory
from src.chat.message import Message
from src.llm.base import LLMProvider
from src.prompt.prompt_builder import PromptBuilder
from src.reranking.reranker import Reranker
from src.retrieval.retrieval_printer import RetrievalPrinter
from src.retrieval.retriever import Retriever


class ChatEngine:

    def __init__(
        self,
        memory: Memory,
        retriever: Retriever,
        reranker: Reranker,
        prompt_builder: PromptBuilder,
        llm: LLMProvider,
    ):
        self.memory = memory
        self.retriever = retriever
        self.reranker = reranker
        self.prompt_builder = prompt_builder
        self.llm = llm

    def ask(
        self,
        question: str,
        top_k: int = 5,
    ) -> Iterator[str]:

        user_message = Message(
            role="user",
            content=question,
        )

        self.memory.add_message(user_message)

        retrieval_top_k = 20

        hybrid_results = self.retriever.retrieve(
            question,
            top_k=retrieval_top_k,
        )

        RetrievalPrinter.print_results(
            "Hybrid Retrieval",
            hybrid_results,
        )

        final_results = hybrid_results

        if self.reranker is not None:

            final_results = self.reranker.rerank(
                query=question,
                results=hybrid_results,
                top_k=top_k,
            )

            RetrievalPrinter.print_results(
                "After CrossEncoder Reranking",
                final_results,
            )

        prompt = self.prompt_builder.build(
            question=question,
            history=self.memory.get_messages(),
            context=final_results,
        )

        response = ""

        for token in self.llm.stream(prompt):
            response += token
            yield token

        assistant_message = Message(
            role="assistant",
            content=response,
        )

        self.memory.add_message(assistant_message)
