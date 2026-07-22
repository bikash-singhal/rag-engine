from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    retrieval_top_k: int
    final_top_k: int
    max_context_tokens: int
    strategy: str
