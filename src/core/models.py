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
