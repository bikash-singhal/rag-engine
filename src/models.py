from dataclasses import dataclass

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