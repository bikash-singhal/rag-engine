from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

import numpy as np

from src.core.latency import LatencyReport


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
class ChatResult:
    """
    Final response produced by the RAG pipeline.
    """

    question: str
    rewritten_question: str
    answer: str
    retrieved_chunks: list[SearchResult]
    latency: LatencyReport


@dataclass(slots=True)
class PreparedPrompt:
    prompt: str
    rewritten_question: str
    retrieved_chunks: list[SearchResult]
    latency: LatencyReport


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


@dataclass(slots=True, frozen=True)
class BenchmarkChunk:
    document: str
    page: int
    chunk_index: int


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass(slots=True, frozen=True)
class BenchmarkCase:
    id: str
    question: str
    expected_chunks: list[BenchmarkChunk]
    expected_answer: str | None = None
    tags: list[str] | None = None
    difficulty: Difficulty = Difficulty.EASY


@dataclass(slots=True, frozen=True)
class BenchmarkDataset:
    name: str
    version: str
    description: str
    created_at: str

    cases: list[BenchmarkCase]


@dataclass(slots=True, frozen=True)
class BenchmarkResult:

    case_id: str
    question: str
    retrieved_chunks: list[SearchResult]
    metric_scores: dict[str, float]
    passed: bool


@dataclass(slots=True, frozen=True)
class BenchmarkSummary:
    experiment_name: str
    benchmark_name: str
    total_cases: int
    metric_scores: dict[str, float]
    passed_cases: int
    failed_cases: int
    results: list[BenchmarkResult]


class JobStatus(str, Enum):
    """
    Lifecycle of an ingestion job.
    """

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestJob:
    """
    Tracks one ingestion task.
    """

    def __init__(
        self,
        filename: str,
    ) -> None:

        self.id = str(uuid4())

        self.filename = filename

        self.status = JobStatus.QUEUED

        self.created_at = datetime.now(UTC)

        self.error: str | None = None
