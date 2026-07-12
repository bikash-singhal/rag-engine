from textwrap import dedent

from src.models import SearchResult


class PromptBuilder:
    """
    Builds prompts for the LLM using retrieved search results.
    """

    def build(
        self,
        question: str,
        results: list[SearchResult],
    ) -> str:
        """
        Builds an LLM prompt from retrieved context.

        Args:
            question: User's question.
            results: Retrieved search results.

        Returns:
            A formatted prompt string.
        """

        if not results:
            context = "No relevant context was retrieved."

        else:
            contexts = []

            for i, result in enumerate(results, start=1):

                contexts.append(
                    dedent(
                        f"""
                        Context {i}
                        Source: {result.chunk.source}

                        {result.chunk.text}
                        """
                    ).strip()
                )

            context = "\n\n-----------------------------\n\n".join(
                contexts
            )

        prompt = dedent(
            f"""
            You are a helpful AI assistant.

            Answer the user's question using ONLY the provided context.

            Rules:
            1. Do not use outside knowledge.
            2. If the answer is not present in the context, say:
               "I don't have enough information to answer that from the provided documents."

            ======================
            CONTEXT
            ======================

            {context}

            ======================
            QUESTION
            ======================

            {question}

            ======================
            ANSWER
            ======================
            """
        ).strip()

        return prompt