from collections.abc import Iterator

from src.chat.memory import Memory
from src.chat.message import Message
from src.config.settings import FINAL_TOP_K, RETRIEVAL_TOP_K
from src.llm.base import LLMProvider
from src.prompt.prompt_builder import PromptBuilder
from src.query.base import QueryRewriter
from src.query.llm_multi_query_generator import LLMMultiQueryGenerator
from src.reranking.reranker import Reranker
from src.retrieval.retrieval_printer import RetrievalPrinter
from src.retrieval.retriever import Retriever
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChatEngine:

    def __init__(
        self,
        memory: Memory,
        query_rewriter: QueryRewriter,
        retriever: Retriever,
        reranker: Reranker,
        multi_query_generator: LLMMultiQueryGenerator,
        prompt_builder: PromptBuilder,
        llm: LLMProvider,
    ):
        self.memory = memory
        self.query_rewriter = query_rewriter
        self.retriever = retriever
        self.reranker = reranker
        self.prompt_builder = prompt_builder
        self.multi_query_generator = multi_query_generator
        self.llm = llm

    def ask(
        self,
        question: str,
    ) -> Iterator[str]:

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

        rewritten_question = self.query_rewriter.rewrite(
            question=question,
            history=self.memory.get_messages(),
        )

        logger.info("Query rewriting completed.")

        logger.debug(
            "Rewritten Question: %s",
            rewritten_question,
        )

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

        retrieval_results = []

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
            len(results),
        )

        RetrievalPrinter.print_results(
            "Hybrid Retrieval",
            retrieval_results,
        )

        final_results = retrieval_results

        logger.info("Starting CrossEncoder reranking.")

        if self.reranker is not None:

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

        logger.info("Building final prompt.")

        prompt = self.prompt_builder.build(
            question=question,
            history=self.memory.get_messages(),
            context=final_results,
        )

        logger.debug(
            "Prompt length: %d characters",
            len(prompt),
        )

        response = ""

        logger.info("Generating answer...")

        for token in self.llm.stream(prompt):
            response += token
            yield token

        logger.info("Answer generation completed.")

        assistant_message = Message(
            role="assistant",
            content=response,
        )

        self.memory.add_message(assistant_message)

        logger.debug(
            "Conversation history size: %d messages",
            len(self.memory.get_messages()),
        )
