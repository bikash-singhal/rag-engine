import re

from src.retrieval.adaptive.retrieval_config import RetrievalConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AdaptiveRetriever:
    """
    Selects retrieval parameters based on query characteristics.

    This first version is deterministic and rule-based. It does not make
    additional LLM calls.
    """

    SIMPLE_CONFIG = RetrievalConfig(
        retrieval_top_k=8,
        final_top_k=4,
        max_context_tokens=2_000,
        strategy="simple",
    )

    COMPARATIVE_CONFIG = RetrievalConfig(
        retrieval_top_k=15,
        final_top_k=6,
        max_context_tokens=3_000,
        strategy="comparative",
    )

    BROAD_CONFIG = RetrievalConfig(
        retrieval_top_k=25,
        final_top_k=8,
        max_context_tokens=4_000,
        strategy="broad",
    )

    DEFAULT_CONFIG = RetrievalConfig(
        retrieval_top_k=12,
        final_top_k=5,
        max_context_tokens=2_500,
        strategy="default",
    )

    _COMPARATIVE_PATTERNS = (
        r"\bdifference between\b",
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bversus\b",
        r"\bvs\.?\b",
        r"\bsimilarities\b",
        r"\badvantages and disadvantages\b",
        r"\bpros and cons\b",
    )

    _BROAD_PATTERNS = (
        r"\bexplain in detail\b",
        r"\bcomprehensive\b",
        r"\beverything about\b",
        r"\bcomplete process\b",
        r"\bcomplete procedure\b",
        r"\bend[- ]to[- ]end\b",
        r"\ball\b",
        r"\boverview\b",
        r"\bsummarize\b",
        r"\bsummary\b",
        r"\bguidelines\b",
        r"\bpolicies\b",
        r"\bprocedures\b",
        r"\bmultiple\b",
    )

    _SIMPLE_STARTERS = (
        "what is",
        "who is",
        "when is",
        "where is",
        "how many",
        "define",
        "state",
        "list the",
    )

    def get_retrieval_config(
        self,
        question: str,
    ) -> RetrievalConfig:
        normalized_question = self._normalize(question)

        if not normalized_question:
            config = self.DEFAULT_CONFIG

        elif self._matches_any(
            normalized_question,
            self._COMPARATIVE_PATTERNS,
        ):
            config = self.COMPARATIVE_CONFIG

        elif self._is_broad(normalized_question):
            config = self.BROAD_CONFIG

        elif self._is_simple(normalized_question):
            config = self.SIMPLE_CONFIG

        else:
            config = self.DEFAULT_CONFIG

        logger.info(
            (
                "Adaptive retrieval strategy=%s, "
                "retrieval_top_k=%d, final_top_k=%d, "
                "max_context_tokens=%d"
            ),
            config.strategy,
            config.retrieval_top_k,
            config.final_top_k,
            config.max_context_tokens,
        )

        return config

    @staticmethod
    def _normalize(question: str) -> str:
        return " ".join(question.lower().strip().split())

    @classmethod
    def _is_simple(cls, question: str) -> bool:
        word_count = len(question.split())

        return word_count <= 12 and question.startswith(cls._SIMPLE_STARTERS)

    @classmethod
    def _is_broad(cls, question: str) -> bool:
        word_count = len(question.split())

        return word_count >= 25 or cls._matches_any(
            question,
            cls._BROAD_PATTERNS,
        )

    @staticmethod
    def _matches_any(
        question: str,
        patterns: tuple[str, ...],
    ) -> bool:
        return any(re.search(pattern, question) for pattern in patterns)
