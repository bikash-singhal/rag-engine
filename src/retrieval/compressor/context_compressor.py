from __future__ import annotations

from src.config.settings import MAX_CONTEXT_TOKENS
from src.core.models import SearchResult
from src.utils.logger import get_logger
from src.utils.timer import timer

logger = get_logger(__name__)


class ContextCompressor:
    """
    Compresses retrieved context before prompt construction.

    Pipeline:
        1. Remove exact duplicate chunks.
        2. Apply context token budget.

    Additional compression strategies (overlap removal, semantic compression,
    LLM compression) can be added later.
    """

    def __init__(
        self,
        max_context_tokens: int = MAX_CONTEXT_TOKENS,
    ) -> None:
        self.max_context_tokens = max_context_tokens

    def compress(
        self,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """
        Compress retrieval results.

        Args:
            results: Ranked retrieval results.

        Returns:
            Compressed retrieval results.
        """

        if not results:
            return []

        input_count = len(results)

        results = self._remove_duplicates(results)

        after_duplicates = len(results)

        results = self._apply_token_budget(results)

        logger.debug(
            ("Context compression: " "%d -> %d chunks " "(duplicates removed=%d)"),
            input_count,
            len(results),
            input_count - after_duplicates,
        )

        return results

    def _remove_duplicates(
        self,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """
        Removes duplicate chunks while preserving ranking.

        Since results are already reranked, the first occurrence is
        considered the highest quality.
        """

        unique_results: list[SearchResult] = []

        seen: set[str] = set()

        for result in results:

            text = result.chunk.text.strip()

            if text in seen:
                continue

            seen.add(text)

            unique_results.append(result)

        return unique_results

    def _apply_token_budget(
        self,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """
        Keeps chunks until the configured token budget is reached.

        Token estimation is intentionally lightweight and model-independent.
        """

        compressed: list[SearchResult] = []

        total_tokens = 0

        for result in results:

            estimated_tokens = len(result.chunk.text.split())

            if total_tokens + estimated_tokens > self.max_context_tokens:
                break

            compressed.append(result)

            total_tokens += estimated_tokens

        logger.debug(
            "Estimated context tokens: %d",
            total_tokens,
        )

        return compressed
