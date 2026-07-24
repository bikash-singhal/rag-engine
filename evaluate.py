import logging

from dotenv import load_dotenv

from src.app.rag_engine import RAGEngine
from src.config.settings import RETRIEVAL_TOP_K
from src.evaluation.benchmark_loader import BenchmarkLoader
from src.evaluation.benchmark_runner import BenchmarkRunner
from src.evaluation.generation_evaluator import GenerationEvaluator
from src.evaluation.generation_metrics import FaithfulnessMetric
from src.evaluation.report_formatter import ReportFormatter
from src.evaluation.retrieval_evaluator import RetrievalEvaluator
from src.evaluation.retrieval_metrics import (
    NDCGMetric,
    RecallAtKMetric,
    ReciprocalRankMetric,
)
from src.utils.logger import get_logger, set_console_log_level

logger = get_logger(__name__)


def main() -> None:

    set_console_log_level(logging.WARNING)

    load_dotenv()

    engine = RAGEngine()

    engine.load_or_ingest(
        document_directory="data/raw",
        index_directory="data/indexes/default",
    )

    dataset = BenchmarkLoader.load(
        "data/benchmark/sagemaker.json",
    )

    retrieval_evaluator = RetrievalEvaluator(
        metrics=[
            RecallAtKMetric(),
            ReciprocalRankMetric(),
            NDCGMetric(),
        ]
    )

    generation_evaluator = GenerationEvaluator(
        metrics=[
            FaithfulnessMetric(),
        ]
    )

    runner = BenchmarkRunner(
        chat_engine=engine.chat_engine,
        retrieval_evaluator=retrieval_evaluator,
        generation_evaluator=generation_evaluator,
    )

    summary = runner.run(
        dataset=dataset,
        experiment_name="Hybrid",
        top_k=RETRIEVAL_TOP_K,
    )

    print(ReportFormatter.format(summary))


if __name__ == "__main__":
    main()
