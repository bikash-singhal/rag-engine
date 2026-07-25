from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from src.core.models import (
    BenchmarkCase,
    BenchmarkChunk,
    BenchmarkDataset,
    SearchResult,
)
from src.retrieval.retriever import Retriever


class BenchmarkBuilder:
    """
    Interactive utility for building benchmark datasets.
    """

    def __init__(
        self,
        retriever: Retriever,
    ) -> None:

        self.retriever = retriever

    def build(
        self,
        questions: list[str],
        benchmark_name: str,
        description: str,
        version: str = "1.0",
        top_k: int = 10,
    ) -> BenchmarkDataset:

        cases: list[BenchmarkCase] = []

        print("=" * 80)
        print("Interactive Benchmark Builder")
        print("=" * 80)

        for index, question in enumerate(
            questions,
            start=1,
        ):

            print(f"\nQuestion:\n{question}\n")

            results = self.retriever.retrieve(
                query=question,
                top_k=top_k,
            )

            print("\nRetrieved Sources:")

            for result in results:
                print(
                    result.chunk.source,
                    result.chunk.page_number,
                    result.chunk.chunk_index,
                )

            self._display_results(results)

            page_input = input("\nGround Truth Page(s) (comma separated): ")

            expected_pages = {
                int(page.strip()) for page in page_input.split(",") if page.strip()
            }

            expected_chunks: list[BenchmarkChunk] = []

            for page in sorted(expected_pages):

                matching_result = next(
                    (result for result in results if result.chunk.page_number == page),
                    None,
                )

                if matching_result is None:

                    document = input(
                        f"Document for page {page} "
                        "(leave blank for current document): "
                    )

                    if not document:
                        document = results[0].chunk.source

                    expected_chunks.append(
                        BenchmarkChunk(
                            document=document,
                            page=page,
                            chunk_index=-1,
                        )
                    )

                    continue

                expected_chunks.append(
                    BenchmarkChunk(
                        document=matching_result.chunk.source,
                        page=page,
                        chunk_index=matching_result.chunk.chunk_index,
                    )
                )

            cases.append(
                BenchmarkCase(
                    id=f"case_{index}",
                    question=question,
                    expected_chunks=expected_chunks,
                    expected_answer=None,
                    tags=[],
                )
            )

            print("✓ Added.")

        return BenchmarkDataset(
            name=benchmark_name,
            version=version,
            description=description,
            created_at=datetime.now().isoformat(),
            cases=cases,
        )

    def save(
        self,
        dataset: BenchmarkDataset,
        output_file: str | Path,
    ) -> None:

        output_file = Path(output_file)

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_file.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                asdict(dataset),
                file,
                indent=4,
                ensure_ascii=False,
            )

    def _display_results(
        self,
        results: list[SearchResult],
    ) -> None:

        print("-" * 80)

        for index, result in enumerate(
            results,
            start=1,
        ):

            print(f"Candidate {index}")
            print(f"Document : {result.chunk.source}")
            print(f"Page     : {result.chunk.page_number}")
            print(f"Chunk    : {result.chunk.chunk_index}")
            print(f"Score    : {result.score:.4f}")
            print(result.chunk.text)
            print("-" * 80)
