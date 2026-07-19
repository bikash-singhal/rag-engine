class MultiQueryPromptBuilder:

    def build(
        self,
        question: str,
    ) -> str:

        return f"""
            You are an expert search assistant.

            Your task is to rewrite the user's search query into equivalent search queries that retrieve the same information from the knowledge base.

            Generate between 1 and 4 equivalent search queries.

            Only generate additional queries if they express the same information need using different wording.

            Rules:

            - Preserve exactly the same user intent.
            - Do NOT broaden the topic.
            - Do NOT narrow the topic.
            - Do NOT introduce related questions.
            - Do NOT introduce new concepts.
            - Prioritize retrieval effectiveness over linguistic variety.
            - Use alternative wording, terminology, entity names, and common aliases.
            - Expand abbreviations and short entity names when helpful.
            - Do NOT answer the question.
            - Do NOT explain anything.
            - Return exactly one query per line.
            - If the original query is already clear and optimal, return only one query.
            - No numbering.
            - No bullets.
            - No blank lines.

            User Question:

            {question}

            Example 1:

            Original:
            What is Amazon SageMaker?

            Good:
            Amazon SageMaker
            What is Amazon SageMaker?
            Amazon SageMaker overview
            Amazon SageMaker definition

            Bad:
            How does Amazon SageMaker work?
            What is Amazon SageMaker used for?
            Amazon SageMaker architecture

            Example 2:

            Original:
            What is RAG?

            Good:
            Retrieval Augmented Generation
            What is Retrieval Augmented Generation?
            RAG overview

            Bad:
            How does RAG work?
            Benefits of RAG
            RAG architecture
            """.strip()
