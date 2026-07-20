from src.config.settings import MAX_RERANK_CANDIDATES
from src.core.models import SearchResult
from src.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.retriever import Retriever


class RerankingRetriever(Retriever):
    """
    Decorates an existing retriever with CrossEncoder reranking.
    """

    def __init__(
        self,
        retriever: Retriever,
        reranker: CrossEncoderReranker,
    ) -> None:

        self.retriever = retriever
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        top_k: int = MAX_RERANK_CANDIDATES,
    ) -> list[SearchResult]:

        results = self.retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        return self.reranker.rerank(
            query=query,
            results=results,
            top_k=top_k,
        )
