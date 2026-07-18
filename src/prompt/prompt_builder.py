from textwrap import dedent

from src.chat.message import Message
from src.core.models import SearchResult


class PromptBuilder:

    def build(
        self,
        question: str,
        history: list[Message],
        context: list[SearchResult],
    ) -> str:

        formatted_history = self._format_history(history)

        formatted_context = self._format_context(context)

        return self._final_prompt(formatted_history, formatted_context, question)

    def _format_history(self, history: list[Message]) -> str:

        if not history:
            return ""

        formatted_history = []

        for message in history:
            formatted_history.append(f"{message.role.title()}:\n")
            formatted_history.append(message.content + "\n\n")

        return "".join(formatted_history)

    def _format_context(self, context: list[SearchResult]) -> str:

        if not context:
            return ""

        formatted_context = []
        _SEPARATOR = "-" * 10

        for index, result in enumerate(context):
            formatted_context.append(f"Context {index + 1}\n")
            formatted_context.append(_SEPARATOR + "\n")
            formatted_context.append(f"Page: {result.chunk.page_number}\n")
            formatted_context.append(f"Chunk: {result.chunk.chunk_index}\n")
            formatted_context.append(result.chunk.text + "\n\n")

        return "".join(formatted_context)

    def _final_prompt(self, history: str, context: str, question: str) -> str:

        prompt = dedent(f"""
            You are a helpful AI assistant.

            Answer the user's question using the retrieved context.

            If the answer cannot be found in the retrieved context, clearly say that you don't know.

            Do not invent information.

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

        return prompt
