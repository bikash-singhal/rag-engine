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
            raise ValueError("chunk_size must be greater than zero.")

        if overlap < 0:
            raise ValueError("overlap cannot be negative.")

        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size.")

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

        chunks: list[Chunk] = []

        chunk_index = 0

        for page in pages:

            words = page.text.split()

            if not words:
                continue

            for start in range(
                0,
                len(words),
                self.step,
            ):

                end = start + self.chunk_size

                chunk_words = words[start:end]

                if not chunk_words:
                    continue

                chunks.append(
                    Chunk(
                        chunk_index=chunk_index,
                        source=page.metadata.get(
                            "source",
                            "",
                        ),
                        page_number=page.page_number,
                        text=" ".join(chunk_words),
                    )
                )

                chunk_index += 1

        return chunks
