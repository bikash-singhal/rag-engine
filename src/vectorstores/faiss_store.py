import faiss
import numpy as np
import pickle

from pathlib import Path

from src.core.models import (
    EmbeddedChunk,
    SearchResult,
)


class FAISSVectorStore:
    """
    Simple in-memory FAISS vector store.

    Note:
        Assumes embeddings are already L2-normalized.
        With normalized vectors, IndexFlatIP performs cosine similarity search.
    """

    def __init__(
        self,
        embedding_dim: int,
    ) -> None:
        """
        Args:
            embedding_dim: Dimension of embedding vectors.
        """

        self.index = faiss.IndexFlatIP(embedding_dim)

        self.chunk_lookup: dict[int, EmbeddedChunk] = {}

    def add(
        self,
        embedded_chunks: list[EmbeddedChunk],
    ) -> None:
        """
        Adds embedded chunks to the FAISS index.
        """

        if not embedded_chunks:
            return

        vectors = np.vstack(
            [
                embedded_chunk.embedding
                for embedded_chunk in embedded_chunks
            ]
        ).astype(np.float32)

        if vectors.shape[1] != self.index.d:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.index.d}, "
                f"got {vectors.shape[1]}."
            )

        start_id = self.index.ntotal

        self.index.add(vectors)

        for offset, embedded_chunk in enumerate(
            embedded_chunks
        ):
            self.chunk_lookup[
                start_id + offset
            ] = embedded_chunk

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Searches the FAISS index for the most similar chunks.
        """

        if self.index.ntotal == 0:
            return []

        query = np.asarray(
            query_embedding,
            dtype=np.float32,
        ).reshape(1, -1)

        if query.shape[1] != self.index.d:
            raise ValueError(
                f"Expected query dimension "
                f"{self.index.d}, "
                f"got {query.shape[1]}."
            )

        scores, indices = self.index.search(
            query,
            top_k,
        )

        results: list[SearchResult] = []

        for score, idx in zip(
            scores[0],
            indices[0],
        ):

            if idx == -1:
                continue

            embedded_chunk = self.chunk_lookup[idx]

            results.append(
                SearchResult(
                    chunk=embedded_chunk.chunk,
                    score=float(score),
                )
            )

        return results
    

    def save(
    self,
    directory: str | Path,
    ) -> None:
        """
        Saves the FAISS index and metadata to disk.
        """

        directory = Path(directory)

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(directory / "faiss.index"),
        )

        with open(
            directory / "metadata.pkl",
            "wb",
        ) as file:

            pickle.dump(
                self.chunk_lookup,
                file,
            )

    @classmethod
    def load(
        cls,
        directory: str | Path,
    ) -> "FAISSVectorStore":
            """
            Loads a previously saved FAISS index.
            """

            directory = Path(directory)

            index = faiss.read_index(
                str(directory / "faiss.index")
            )

            with open(
                directory / "metadata.pkl",
                "rb",
            ) as file:

                chunk_lookup = pickle.load(file)

            vector_store = cls(
                embedding_dim=index.d,
            )

            vector_store.index = index

            vector_store.chunk_lookup = chunk_lookup

            return vector_store