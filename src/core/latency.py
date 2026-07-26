from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LatencyReport:
    """
    Latency measurements for a single RAG pipeline execution.
    """

    query_rewrite_ms: float = 0.0
    multi_query_ms: float = 0.0
    retrieval_ms: float = 0.0
    reranking_ms: float = 0.0
    prompt_build_ms: float = 0.0
    answer_generation_ms: float = 0.0
    total_ms: float = 0.0

    @property
    def stages(self) -> dict[str, float]:
        return {
            "Query Rewrite": self.query_rewrite_ms,
            "Multi Query": self.multi_query_ms,
            "Retrieval": self.retrieval_ms,
            "Reranking": self.reranking_ms,
            "Prompt Build": self.prompt_build_ms,
            "Answer Generation": self.answer_generation_ms,
        }

    @property
    def all_metrics(self) -> dict[str, float]:
        return {
            **self.stages,
            "Total": self.total_ms,
        }

    @property
    def pipeline_ms(self) -> float:
        return (
            self.query_rewrite_ms
            + self.multi_query_ms
            + self.retrieval_ms
            + self.reranking_ms
            + self.prompt_build_ms
            + self.answer_generation_ms
        )

    @property
    def orchestration_ms(self) -> float:
        return max(0.0, self.total_ms - self.pipeline_ms)

    @classmethod
    def average(cls, reports: list[LatencyReport]) -> LatencyReport:
        if not reports:
            return cls()

        n = len(reports)

        return cls(
            query_rewrite_ms=sum(r.query_rewrite_ms for r in reports) / n,
            multi_query_ms=sum(r.multi_query_ms for r in reports) / n,
            retrieval_ms=sum(r.retrieval_ms for r in reports) / n,
            reranking_ms=sum(r.reranking_ms for r in reports) / n,
            prompt_build_ms=sum(r.prompt_build_ms for r in reports) / n,
            answer_generation_ms=sum(r.answer_generation_ms for r in reports) / n,
            total_ms=sum(r.total_ms for r in reports) / n,
        )
