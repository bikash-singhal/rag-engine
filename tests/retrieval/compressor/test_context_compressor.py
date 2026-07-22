from src.core.models import Chunk, SearchResult
from src.retrieval.compressor.context_compressor import ContextCompressor


def make_result(
    text: str,
    score: float = 1.0,
) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_index=0,
            source="test.pdf",
            page_number=1,
            text=text,
        ),
        score=score,
    )


def test_empty_results() -> None:

    compressor = ContextCompressor()

    assert compressor.compress([]) == []


def test_remove_duplicates() -> None:

    compressor = ContextCompressor()

    results = [
        make_result("Hello World", 0.9),
        make_result("Hello World", 0.8),
        make_result("Different Text", 0.7),
    ]

    compressed = compressor.compress(results)

    assert len(compressed) == 2

    assert compressed[0].chunk.text == "Hello World"

    assert compressed[1].chunk.text == "Different Text"


def test_token_budget() -> None:

    compressor = ContextCompressor(
        max_context_tokens=5,
    )

    results = [
        make_result("one two"),
        make_result("three four"),
        make_result("five six"),
    ]

    compressed = compressor.compress(results)

    assert len(compressed) == 2


def test_no_duplicates() -> None:

    compressor = ContextCompressor()

    results = [
        make_result("A"),
        make_result("B"),
        make_result("C"),
    ]

    compressed = compressor.compress(results)

    assert len(compressed) == 3
