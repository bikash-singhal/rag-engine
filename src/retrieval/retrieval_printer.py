from src.core.models import SearchResult


class RetrievalPrinter:

    @staticmethod
    def print_results(
        title: str,
        results: list[SearchResult],
    ) -> None:

        print()

        print("=" * 80)
        print(title)
        print("=" * 80)

        for rank, result in enumerate(results, start=1):

            print(f"Rank        : {rank}")
            print(f"Score       : {result.score:.4f}")
            print(f"Page        : {result.chunk.page_number}")
            print(f"Chunk Index : {result.chunk.chunk_index}")
            print(result.chunk.text)
            print("-" * 80)
