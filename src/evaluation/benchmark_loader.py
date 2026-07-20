import json
from pathlib import Path

from src.core.models import BenchmarkCase, BenchmarkChunk, BenchmarkDataset
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BenchmarkLoader:
    """
    Loads a benchmark dataset from JSON into strongly-typed models.
    """

    @staticmethod
    def load(
        path: str | Path,
    ) -> BenchmarkDataset:

        path = Path(path)

        logger.info("Loading benchmark dataset: %s", path)

        raw_data = BenchmarkLoader._read_json(path)

        benchmark_name = raw_data.get("name", "Unnamed Benchmark")
        description = raw_data.get("description", "")

        version = raw_data.get("version")
        created_at = raw_data.get("created_at")

        if version is None:
            raise ValueError("Benchmark metadata is missing 'version'.")

        if created_at is None:
            raise ValueError("Benchmark metadata is missing 'created_at'.")

        raw_cases = raw_data.get("cases")

        if raw_cases is None:
            raise ValueError("Benchmark file is missing 'cases' section.")

        cases = [BenchmarkLoader._parse_case(case) for case in raw_cases]

        logger.info(
            "Loaded %d benchmark cases.",
            len(cases),
        )

        return BenchmarkDataset(
            name=benchmark_name,
            version=version,
            description=description,
            created_at=created_at,
            cases=cases,
        )

    @staticmethod
    def _read_json(
        path: Path,
    ) -> dict:

        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {path}")

        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)

    @staticmethod
    def _parse_case(
        data: dict,
    ) -> BenchmarkCase:

        required_fields = (
            "id",
            "question",
            "expected_chunks",
        )

        for field in required_fields:

            if field not in data:

                raise ValueError(f"Benchmark case is missing required field '{field}'.")

        expected_chunks = [
            BenchmarkLoader._parse_chunk(chunk) for chunk in data["expected_chunks"]
        ]

        return BenchmarkCase(
            id=data["id"],
            question=data["question"],
            expected_chunks=expected_chunks,
            expected_answer=data.get("expected_answer"),
            tags=data.get("tags", []),
        )

    @staticmethod
    def _parse_chunk(
        data: dict,
    ) -> BenchmarkChunk:

        required_fields = (
            "document",
            "page",
            "chunk_index",
        )

        for field in required_fields:

            if field not in data:

                raise ValueError(f"Expected chunk is missing required field '{field}'.")

        return BenchmarkChunk(
            document=data["document"],
            page=data["page"],
            chunk_index=data.get("chunk_index", -1),
        )
