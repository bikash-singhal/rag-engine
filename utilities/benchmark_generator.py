from src.core.models import SearchResult
from src.retrieval.retriever import Retriever


class BenchmarkGenerator:
    """
    Interactive utility to build benchmark datasets.

    For each question:
      - Retrieves Top-K chunks
      - Displays them
      - Prompts user for ground-truth chunk IDs
      - Produces benchmark tuples
    """

    def __init__(
        self,
        retriever: Retriever,
    ) -> None:

        self.retriever = retriever

    def build(
        self,
        questions: list[str],
        top_k: int = 10,
    ) -> list[tuple[str, set[int]]]:

        benchmark: list[tuple[str, set[int]]] = []

        print("=" * 80)
        print("Interactive Benchmark Builder")
        print("=" * 80)

        for question in questions:

            print(f"\nQuestion:\n{question}\n")

            results = self.retriever.retrieve(
                query=question,
                top_k=top_k,
            )

            self._display_results(results)

            chunk_input = input("\nGround Truth Chunk(s) (comma separated): ")

            expected_chunk_ids = {
                int(chunk.strip()) for chunk in chunk_input.split(",") if chunk.strip()
            }

            benchmark.append(
                (
                    question,
                    expected_chunk_ids,
                )
            )

            print("✓ Added.")

        return benchmark

    def _display_results(
        self,
        results: list[SearchResult],
    ) -> None:

        print("-" * 80)

        for i, result in enumerate(results, start=1):

            print(f"Candidate {i}")
            print(f"Chunk : {result.chunk.chunk_index}")
            print(f"Page  : {result.chunk.page_number}")
            print(f"Score : {result.score:.4f}")
            print(result.chunk.text)
            print("-" * 80)
