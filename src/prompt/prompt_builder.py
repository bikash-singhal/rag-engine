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
            formatted_context.append(result.chunk.text.strip())
            formatted_context.append(f"\n\n{separator}\n\n")

        return "".join(formatted_context)

    def _final_prompt(self, history: str, context: str, question: str) -> str:

        prompt = dedent(f"""
            You are a helpful AI assistant.

            Treat the retrieved context as the single source of truth.

            If the answer cannot be found in the retrieved context, say:

            "I don't know based on the provided documents."

            Do not invent facts.
            Do not use outside knowledge.

            Whenever you use information from a retrieved context,
            cite ONLY the source(s) that directly contain that information.

            Do not cite every retrieved source.

            Do not guess citations.

            If only one source supports a sentence,
            cite exactly one source.

            Use citations in the format:

            [Source X]

            where X is the source number shown in the retrieved context. 
            If multiple sources support the same sentence, cite all of them.
            Never cite a source that was not provided.

            Examples:

            [Source 1]
            [Source 3]
            [Source 2][Source 5]

            Only cite the source(s) that directly support the statement.

            Do not cite every retrieved source.

            Do not invent source numbers.
                        
            Example:

            Amazon SageMaker is a fully managed machine learning service. [Source 1]

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
