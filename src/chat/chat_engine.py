from collections.abc import Iterator
from time import perf_counter
from typing import TYPE_CHECKING

from src.chat.memory import Memory
from src.chat.message import Message
from src.core.latency import LatencyReport
from src.core.models import ChatResult, PreparedPrompt, SearchResult
from src.evaluation.latency_printer import LatencyPrinter
from src.llm.base import LLMProvider
from src.prompt.prompt_builder import PromptBuilder
from src.query.base import QueryRewriter
from src.query.llm_multi_query_generator import LLMMultiQueryGenerator
from src.reranking.reranker import Reranker
from src.retrieval.adaptive import AdaptiveRetriever
from src.retrieval.compressor.context_compressor import ContextCompressor
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
        adaptive_retriever: AdaptiveRetriever,
        multi_query_generator: LLMMultiQueryGenerator,
        prompt_builder: PromptBuilder,
        llm: LLMProvider,
    ):
        self.memory = memory
        self.query_rewriter = query_rewriter
        self.retriever = retriever
        self.reranker = reranker
        self.context_compressor = context_compressor
        self.adaptive_retriever = adaptive_retriever
        self.prompt_builder = prompt_builder
        self.multi_query_generator = multi_query_generator
        self.llm = llm

    def _add_user_message(
        self,
        question: str,
    ) -> None:

        logger.debug("Processing user question.")
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

        retrieval_config = self.adaptive_retriever.get_retrieval_config(
            question,
        )

        logger.debug("Query rewriting started.")
        with timer(latency, "query_rewrite_ms"):
            rewritten_question = self.query_rewriter.rewrite(
                question=question,
                history=self.memory.get_messages(),
            )

        logger.debug("Query rewriting completed.")

        logger.debug(
            "Rewritten Question: %s",
            rewritten_question,
        )

        with timer(latency, "multi_query_ms"):
            queries = self.multi_query_generator.generate(rewritten_question)

        logger.debug(
            "Generated %d retrieval queries.",
            len(queries),
        )

        for i, query in enumerate(queries, start=1):
            logger.debug(
                "Query %d: %s",
                i,
                query,
            )

        logger.debug("Starting retrieval.")

        with timer(latency, "retrieval_ms"):
            retrieval_results: list[SearchResult] = []

            for query in queries:
                logger.debug(
                    "Retrieving with query: %s",
                    query,
                )

                results = self.retriever.retrieve(
                    query,
                    top_k=retrieval_config.retrieval_top_k,
                )

                retrieval_results.extend(results)

        logger.debug(
            "Retrieved %d candidates.",
            len(retrieval_results),
        )

        final_results = retrieval_results

        logger.debug("Starting CrossEncoder reranking.")

        with timer(latency, "reranking_ms"):
            final_results = self.reranker.rerank(
                query=rewritten_question,
                results=retrieval_results,
                top_k=retrieval_config.final_top_k,
            )

        logger.debug(
            "Top %d candidates selected after reranking.",
            len(final_results),
        )

        final_results = self.context_compressor.compress(
            final_results,
            max_context_tokens=retrieval_config.max_context_tokens,
        )

        logger.debug("Building final prompt.")
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

        logger.info(
            "Answer generated in %.1f ms",
            prepared.latency.answer_generation_ms,
        )

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

    def evaluate(
        self,
        question: str,
    ) -> ChatResult:

        overall_start = perf_counter()

        prepared = self._prepare_prompt(question)

        logger.debug("Generating answer...")
        with timer(prepared.latency, "answer_generation_ms"):
            answer = self.llm.generate(prepared.prompt)

        logger.debug("Answer generation completed.")

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

        # TODO: Capture streaming-specific latency metrics
        # (TTFT, stream duration, total latency).

        self._add_user_message(question)

        prepared = self._prepare_prompt(question)

        logger.debug("Generating answer...")

        response = ""

        for token in self.llm.stream(prepared.prompt):
            response += token
            yield token

        logger.debug("Answer generation completed.")

        self._add_assistant_message(response)
