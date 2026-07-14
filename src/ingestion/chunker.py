from collections.abc import Iterator

from src.core.models import Chunk, Page


class Chunker:
    """
    Splits streamed pages into overlapping word-based chunks.
    """

    def __init__(
        self,
        chunk_size: int,
        overlap: int,
    ) -> None:

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero."
            )

        if overlap < 0:
            raise ValueError(
                "overlap cannot be negative."
            )

        if overlap >= chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk_size."
            )

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.step = chunk_size - overlap

    def chunk(
        self,
        pages: Iterator[Page],
    ) -> list[Chunk]:
        """
        Splits streamed pages into overlapping chunks.
        """

        full_text = "\n".join(
            page.text
            for page in pages
        )

        words = full_text.split()

        if not words:
            return []

        chunks: list[Chunk] = []

        for chunk_index, start in enumerate(
            range(0, len(words), self.step)
        ):

            end = start + self.chunk_size

            chunk_words = words[start:end]

            text = " ".join(chunk_words)

            if not text:
                continue

            chunks.append(
                Chunk(
                    chunk_index=chunk_index,
                    source="",
                    text=text,
                )
            )

        return chunks