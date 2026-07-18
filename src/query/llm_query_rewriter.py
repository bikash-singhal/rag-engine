from src.chat.message import Message
from src.llm.base import LLMProvider
from src.query.base import QueryRewriter
from src.query.rewrite_prompt import RewritePromptBuilder


class LLMQueryRewriter(QueryRewriter):

    def __init__(
        self,
        llm: LLMProvider,
        prompt_builder: RewritePromptBuilder,
    ):
        self.llm = llm
        self.prompt_builder = prompt_builder

    def rewrite(
        self,
        question: str,
        history: list[Message],
    ) -> str:

        prompt = self.prompt_builder.build(
            question=question,
            history=history,
        )

        rewritten_query = self.llm.generate(prompt)

        return rewritten_query.strip()
