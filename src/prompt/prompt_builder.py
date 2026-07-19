from textwrap import dedent

from src.chat.message import Message
from src.core.models import SearchResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PromptBuilder:

    def build(
        self,
        question: str,
        history: list[Message],
        context: list[SearchResult],
    ) -> str:

        formatted_history = self._format_history(history)

        formatted_context = self._format_context(context)

        logger.debug(
            "History messages: %d | Context chunks: %d",
            len(history),
            len(context),
        )

        return self._final_prompt(formatted_history, formatted_context, question)

    def _format_history(self, history: list[Message]) -> str:

        if not history:
            return ""

        formatted_history = []

        for message in history:
            formatted_history.append(f"{message.role.title()}:\n")
            formatted_history.append(message.content + "\n\n")

        return "".join(formatted_history)

    def _format_context(
        self,
        context: list[SearchResult],
    ) -> str:

        if not context:
            return ""

        formatted_context = []

        separator = "-" * 20

        for index, result in enumerate(context, start=1):

            formatted_context.append(f"[Source {index}]\n")
            formatted_context.append(f"Document: {result.chunk.source}\n")
            formatted_context.append(f"Page: {result.chunk.page_number}\n")
            formatted_context.append(f"Chunk: {result.chunk.chunk_index}\n\n")
            formatted_context.append(result.chunk.text)
            formatted_context.append(f"\n\n{separator}\n\n")

        return "".join(formatted_context)

    def _final_prompt(self, history: str, context: str, question: str) -> str:

        prompt = dedent(f"""
            You are a helpful AI assistant.

            Treat the retrieved context as the ONLY source of truth.

            If the answer cannot be found in the retrieved context, respond with exactly:

            "I don't know based on the provided documents."

            Do not invent facts.
            Do not use outside knowledge.
            Do not make assumptions.

            ======================
            Citation Rules
            ======================

            - Cite only the source(s) that directly support a statement.
            - Prefer the highest-ranked source whenever possible.
            - Prefer the smallest number of sources needed.
            - Do NOT cite sources that merely repeat the same information.
            - Never invent source numbers.
            - Never cite a source that was not provided.

            Use citations in the format:

            [Source X]

            Examples:

            Good:

            Amazon SageMaker is a fully managed machine learning service. [Source 1]

            Amazon SageMaker supports SQL analytics and generative AI capabilities. [Source 2][Source 4]

            Bad:

            Amazon SageMaker is a fully managed machine learning service. [Source 1][Source 2][Source 3][Source 4][Source 5]

            Amazon SageMaker is a fully managed machine learning service. [Source 8]

            Amazon SageMaker is a fully managed machine learning service.
            (no citation)

            ========================
            Previous Conversation
            ========================

            {history}

            ======================
            RETRIEVED CONTEXT
            ======================

            {context}

            ======================
            CURRENT QUESTION
            ======================

            {question}

            ======================
            ANSWER
            ======================
            """).strip()

        logger.debug(
            "Prompt length: %d characters",
            len(prompt),
        )

        return prompt
