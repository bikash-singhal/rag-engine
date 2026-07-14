import numpy as np
from sentence_transformers import SentenceTransformer

from src.core.models import Chunk, EmbeddedChunk


class Embedder:
    """
    Generates vector embeddings for document chunks using a
    Sentence Transformer model.
    """

    def __init__(
        self,
        model_name: str,
    ) -> None:
        """
        Args:
            model_name: Hugging Face Sentence Transformer model.
        """

        self.model = SentenceTransformer(model_name)

        self.embedding_dimension = (
            self.model.get_embedding_dimension()
        )

    def embed(
        self,
        chunks: list[Chunk],
    ) -> list[EmbeddedChunk]:
        """
        Generates embeddings for a list of chunks.

        Args:
            chunks: List of Chunk objects.

        Returns:
            List of EmbeddedChunk objects.
        """

        if not chunks:
            return []

        texts = [
            chunk.text
            for chunk in chunks
        ]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        embedded_chunks = []

        for chunk, embedding in zip(
            chunks,
            embeddings,
        ):
            embedded_chunks.append(
                EmbeddedChunk(
                    chunk=chunk,
                    embedding=embedding,
                )
            )

        return embedded_chunks
    

    def embed_query(
    self,
    query: str,
    ) -> np.ndarray:
        
        """
        Generates an embedding for a user query.
        """

        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embedding.astype(np.float32)