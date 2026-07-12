import faiss
import numpy as np

from src.models import EmbeddedChunk, SearchResult


class FAISSVectorStore:
    """
    A simple in-memory FAISS vector store using cosine similarity
    (implemented as Inner Product on normalized vectors).
    """

    def __init__(
        self,
        embedding_dim: int,
    ) -> None:
        """
        Args:
            embedding_dim: Dimension of the embedding vectors.
        """

        self.index = faiss.IndexFlatIP(embedding_dim)

        self.id_to_chunk: dict[int, EmbeddedChunk] = {}

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
            [chunk.embedding for chunk in embedded_chunks]
        ).astype(np.float32)

        if vectors.shape[1] != self.index.d:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.index.d}, "
                f"got {vectors.shape[1]}."
            )

        start_id = self.index.ntotal

        self.index.add(vectors)

        for offset, embedded_chunk in enumerate(embedded_chunks):
            self.id_to_chunk[start_id + offset] = embedded_chunk

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

        query = np.expand_dims(
            query_embedding,
            axis=0,
        ).astype(np.float32)

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

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            embedded_chunk = self.id_to_chunk[idx]

            results.append(
                SearchResult(
                    chunk=embedded_chunk.chunk,
                    score=float(score),
                )
            )

        return results