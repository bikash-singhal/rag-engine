from textwrap import dedent

from src.chat.message import Message


class RewritePromptBuilder:

    def __init__(
        self,
        history_limit: int = 4,
    ):
        self.history_limit = history_limit

    def build(
        self,
        question: str,
        history: list[Message],
    ) -> str:

        formatted_history = self._format_history(history)

        return dedent(f"""
        You are an AI assistant that rewrites follow-up questions
        into standalone search queries.

        Your task is ONLY to rewrite the user's question.

        If the current question is already complete and unambiguous,
        return it unchanged.

        Preserve the user's original intent.

        Resolve references such as:

        - it
        - they
        - this
        - that
        - those

        Do not answer the question.

        Return ONLY the rewritten query.

        ========================
        Conversation History
        ========================

        {formatted_history}

        ========================
        Current Question
        ========================

        {question}

        ========================
        Rewritten Query
        ========================
        """).strip()

    def _format_history(self, history: list[Message]) -> str:

        if not history:
            return ""

        formatted_history = []

        for message in history[-self.history_limit :]:
            formatted_history.append(
                f"{message.role.title()}:\n" f"{message.content}\n\n"
            )

        return "".join(formatted_history)
