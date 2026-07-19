from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class Document:
    """
    Represents a complete document.
    """

    source: str
    text: str


@dataclass(slots=True)
class Chunk:
    """
    Represents one chunk of a document.
    """

    chunk_index: int
    source: str
    page_number: int
    text: str

    @property
    def id(self) -> tuple[str, int]:
        return (
            self.source,
            self.chunk_index,
        )


@dataclass(slots=True)
class EmbeddedChunk:
    """
    Chunk together with its embedding vector.
    """

    chunk: Chunk
    embedding: np.ndarray


@dataclass(slots=True)
class SearchResult:
    """
    Represents one retrieval result.
    """

    chunk: Chunk
    score: float


@dataclass(slots=True)
class Page:
    """
    Represents one page of a document.
    """

    page_number: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalStatistics:
    """
    Statistics describing a retrieval operation.
    """

    result_count: int
    highest_score: float
    lowest_score: float
    average_score: float
    unique_pages: list[int]
    page_diversity: float


@dataclass(slots=True)
class RetrievalReport:
    """
    Complete retrieval report.
    """

    question: str
    results: list[SearchResult]
    statistics: RetrievalStatistics


@dataclass
class RetrievalEvaluation:
    hit_rate: float
    precision: float
    recall: float
    f1_score: float
    mrr: float


@dataclass(slots=True)
class BenchmarkEvaluation:
    average_hit_rate: float
    average_precision: float
    average_recall: float
    average_f1_score: float
    average_mrr: float


@dataclass(slots=True)
class ExperimentResult:
    name: str
    benchmark: BenchmarkEvaluation


@dataclass(slots=True)
class ExperimentReport:
    experiments: list[ExperimentResult]
