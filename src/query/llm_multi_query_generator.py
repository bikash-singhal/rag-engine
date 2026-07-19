from src.config.settings import NUM_MULTI_QUERIES
from src.llm.base import LLMProvider
from src.prompt.multi_query_prompt_builder import MultiQueryPromptBuilder
from src.query.multi_query_generator import MultiQueryGenerator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMMultiQueryGenerator(MultiQueryGenerator):

    def __init__(
        self,
        llm: LLMProvider,
    ):
        self.llm = llm
        self.prompt_builder = MultiQueryPromptBuilder()

    def generate(
        self,
        question: str,
    ) -> list[str]:

        prompt = self.prompt_builder.build(question)

        response = self.llm.generate(prompt)

        queries = [line.strip() for line in response.splitlines() if line.strip()]

        queries = [question] + queries

        queries = list(dict.fromkeys(queries))

        if not queries:
            return [question]

        queries = queries[:NUM_MULTI_QUERIES]

        logger.debug(
            "Multi-query generation produced %d unique queries.",
            len(queries),
        )

        return queries
