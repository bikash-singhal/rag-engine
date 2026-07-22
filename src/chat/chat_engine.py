from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter

from src.chat.memory import Memory
from src.chat.message import Message
from src.config.settings import FINAL_TOP_K, RETRIEVAL_TOP_K
from src.core.latency import LatencyReport
from src.core.models import ChatResult, PreparedPrompt, SearchResult
from src.evaluation.latency_printer import LatencyPrinter
from src.llm.base import LLMProvider
from src.prompt.prompt_builder import PromptBuilder
from src.query.base import QueryRewriter
from src.query.llm_multi_query_generator import LLMMultiQueryGenerator
from src.reranking.reranker import Reranker
from src.retrieval.compressor.context_compressor import ContextCompressor
from src.retrieval.retrieval_printer import RetrievalPrinter
from src.retrieval.retriever import Retriever
from src.utils.logger import get_logger
from src.utils.timer import timer

logger = get_logger(__name__)


class ChatEngine:

    def __init__(
        self,
        memory: Memory,
        query_rewriter: QueryRewriter,
        retriever: Retriever,
        reranker: Reranker,
        context_compressor: ContextCompressor,
        multi_query_generator: LLMMultiQueryGenerator,
        prompt_builder: PromptBuilder,
        llm: LLMProvider,
    ):
        self.memory = memory
        self.query_rewriter = query_rewriter
        self.retriever = retriever
        self.reranker = reranker
        self.context_compressor = context_compressor
        self.prompt_builder = prompt_builder
        self.multi_query_generator = multi_query_generator
        self.llm = llm

    def _add_user_message(
        self,
        question: str,
    ) -> None:

        logger.info("Processing user question.")
        logger.debug("Original Question: %s", question)

        user_message = Message(
            role="user",
            content=question,
        )

        self.memory.add_message(user_message)

        logger.debug(
            "Conversation history size: %d messages",
            len(self.memory.get_messages()),
        )

    def _add_assistant_message(self, answer: str) -> None:

        assistant_message = Message(
            role="assistant",
            content=answer,
        )

        self.memory.add_message(assistant_message)

    def _prepare_prompt(
        self,
        question: str,
    ) -> PreparedPrompt:

        latency = LatencyReport()
        logger.info("Query rewriting started.")
        with timer(latency, "query_rewrite_ms"):
            rewritten_question = self.query_rewriter.rewrite(
                question=question,
                history=self.memory.get_messages(),
            )

        logger.info("Query rewriting completed.")

        logger.debug(
            "Rewritten Question: %s",
            rewritten_question,
        )

        with timer(latency, "multi_query_ms"):
            queries = self.multi_query_generator.generate(rewritten_question)

        logger.info(
            "Generated %d retrieval queries.",
            len(queries),
        )

        for i, query in enumerate(queries, start=1):
            logger.debug(
                "Query %d: %s",
                i,
                query,
            )

        logger.info("Starting retrieval.")

        with timer(latency, "retrieval_ms"):
            retrieval_results: list[SearchResult] = []

            for query in queries:
                logger.debug(
                    "Retrieving with query: %s",
                    query,
                )

                results = self.retriever.retrieve(
                    query,
                    top_k=RETRIEVAL_TOP_K,
                )

                retrieval_results.extend(results)

        logger.debug(
            "Retrieved %d candidates.",
            len(retrieval_results),
        )

        RetrievalPrinter.print_results(
            "Hybrid Retrieval",
            retrieval_results,
        )

        final_results = retrieval_results

        logger.info("Starting CrossEncoder reranking.")

        if self.reranker is not None:
            with timer(latency, "reranking_ms"):
                final_results = self.reranker.rerank(
                    query=rewritten_question,
                    results=retrieval_results,
                    top_k=FINAL_TOP_K,
                )

                RetrievalPrinter.print_results(
                    "After CrossEncoder Reranking",
                    final_results,
                )

        logger.info(
            "Top %d candidates selected after reranking.",
            len(final_results),
        )

        final_results = self.context_compressor.compress(final_results)

        logger.info("Building final prompt.")
        with timer(latency, "prompt_build_ms"):
            prompt = self.prompt_builder.build(
                question=question,
                history=self.memory.get_messages(),
                context=final_results,
            )

        logger.debug(
            "Prompt length: %d characters",
            len(prompt),
        )

        return PreparedPrompt(
            prompt=prompt,
            rewritten_question=rewritten_question,
            retrieved_chunks=final_results,
            latency=latency,
        )

    def chat(
        self,
        question: str,
    ) -> ChatResult:

        overall_start = perf_counter()

        self._add_user_message(question)

        prepared = self._prepare_prompt(question)

        logger.info("Generating answer...")
        with timer(prepared.latency, "answer_generation_ms"):
            answer = self.llm.generate(prepared.prompt)

        logger.info("Answer generation completed.")

        self._add_assistant_message(answer)

        prepared.latency.total_ms = (perf_counter() - overall_start) * 1000

        LatencyPrinter.print(prepared.latency)

        return ChatResult(
            question=question,
            rewritten_question=prepared.rewritten_question,
            answer=answer,
            retrieved_chunks=prepared.retrieved_chunks,
            latency=prepared.latency,
        )

    def ask(
        self,
        question: str,
    ) -> Iterator[str]:

        self._add_user_message(question)

        prepared = self._prepare_prompt(question)

        logger.info("Generating answer...")

        response = ""

        for token in self.llm.stream(prepared.prompt):
            response += token
            yield token

        logger.info("Answer generation completed.")

        self._add_assistant_message(response)
