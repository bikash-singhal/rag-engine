from textwrap import dedent

from src.chat.message import Message


class RewritePromptBuilder:

    def build(
        self,
        question: str,
        history: str,
    ) -> str:

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

        {history}

        ========================
        Current Question
        ========================

        {question}

        ========================
        Rewritten Query
        ========================
        """).strip()
