from __future__ import annotations

import math
from abc import ABC, abstractmethod

from src.core.models import BenchmarkCase, SearchResult


class RetrievalMetric(ABC):
    """
    Base interface for all retrieval metrics.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def compute(
        self,
        case: BenchmarkCase,
        retrieved: list[SearchResult],
    ) -> float: ...


def _is_relevant(
    case: BenchmarkCase,
    result: SearchResult,
) -> bool:
    """
    Returns True if the retrieved chunk matches one of the expected
    benchmark chunks.

    Current matching strategy:
        document + page
    """

    for expected in case.expected_chunks:

        if (
            expected.document == result.chunk.source
            and expected.page == result.chunk.page_number
        ):
            return True

    return False


class RecallAtKMetric(RetrievalMetric):

    @property
    def name(self) -> str:
        return "Recall@K"

    def compute(
        self,
        case: BenchmarkCase,
        retrieved: list[SearchResult],
    ) -> float:

        expected_pages = {
            (chunk.document, chunk.page) for chunk in case.expected_chunks
        }

        if not expected_pages:
            return 0.0

        retrieved_pages = {
            (result.chunk.source, result.chunk.page_number)
            for result in retrieved
            if _is_relevant(case, result)
        }

        hits = expected_pages & retrieved_pages

        return len(hits) / len(expected_pages)


class ReciprocalRankMetric(RetrievalMetric):

    @property
    def name(self) -> str:
        return "MRR"

    def compute(
        self,
        case: BenchmarkCase,
        retrieved: list[SearchResult],
    ) -> float:

        for rank, result in enumerate(
            retrieved,
            start=1,
        ):

            if _is_relevant(case, result):
                return 1.0 / rank

        return 0.0


class NDCGMetric(RetrievalMetric):

    @property
    def name(self) -> str:
        return "nDCG"

    def compute(
        self,
        case: BenchmarkCase,
        retrieved: list[SearchResult],
    ) -> float:

        dcg = 0.0

        for rank, result in enumerate(
            retrieved,
            start=1,
        ):

            if _is_relevant(case, result):

                dcg += 1.0 / math.log2(rank + 1)

        ideal_hits = min(
            len(case.expected_chunks),
            len(retrieved),
        )

        if ideal_hits == 0:
            return 0.0

        idcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(
                1,
                ideal_hits + 1,
            )
        )

        return dcg / idcg
