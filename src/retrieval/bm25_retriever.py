import numpy as np
from rank_bm25 import BM25Okapi

from src.core.models import Chunk, SearchResult
from src.retrieval.retriever import Retriever
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BM25Retriever(Retriever):

    def __init__(
        self,
        chunks: list[Chunk],
    ) -> None:

        if not chunks:
            raise ValueError("Chunks cannot be empty.")

        self.chunks = chunks

        corpus = [self._tokenize(chunk.text) for chunk in chunks]

        self.bm25 = BM25Okapi(corpus)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:

        logger.debug("BM25 retrieval started.")

        query_tokens = self._tokenize(query)

        scores = self.bm25.get_scores(query_tokens)

        top_k = min(
            top_k,
            len(self.chunks),
        )

        top_indices = np.argpartition(
            scores,
            -top_k,
        )[-top_k:]

        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results: list[SearchResult] = []

        for index in top_indices:

            results.append(
                SearchResult(
                    chunk=self.chunks[index],
                    score=float(scores[index]),
                )
            )

        logger.debug(
            "Dense retrieval returned %d results.",
            len(results),
        )

        return results

    def _tokenize(
        self,
        text: str,
    ) -> list[str]:

        return text.lower().split()
