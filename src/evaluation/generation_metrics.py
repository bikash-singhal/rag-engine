from src.utils.logger import get_logger

logger = get_logger(__name__)
import time
from abc import ABC, abstractmethod

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness

from src.evaluation.ragas_llm import create_ragas_llm


class GenerationMetric(ABC):

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def evaluate(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        expected_answer: str | None = None,
    ) -> float: ...


class FaithfulnessMetric(GenerationMetric):

    def __init__(
        self,
        max_attempts: int = 3,
        retry_delay_seconds: float = 1.0,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")

        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds cannot be negative.")

        self._metric = Faithfulness(
            llm=create_ragas_llm(),
        )
        self._max_attempts = max_attempts
        self._retry_delay_seconds = retry_delay_seconds

    @property
    def name(self) -> str:
        return "faithfulness"

    def evaluate(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        expected_answer: str | None = None,
    ) -> float:

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )

        for attempt in range(1, self._max_attempts + 1):
            try:
                score = float(self._metric.single_turn_score(sample))

                if attempt > 1:
                    logger.info(
                        "Faithfulness evaluation succeeded after retry. "
                        "question=%r attempt=%d",
                        question,
                        attempt,
                    )

                return score

            except Exception as exc:
                is_final_attempt = attempt == self._max_attempts

                logger.warning(
                    "Faithfulness evaluation attempt failed. "
                    "question=%r attempt=%d max_attempts=%d "
                    "error_type=%s error=%s",
                    question,
                    attempt,
                    self._max_attempts,
                    type(exc).__name__,
                    exc,
                    exc_info=is_final_attempt,
                )

                if is_final_attempt:
                    raise RuntimeError(
                        "Faithfulness evaluation failed after "
                        f"{self._max_attempts} attempts for "
                        f"question: {question!r}"
                    ) from exc

                if self._retry_delay_seconds > 0:
                    time.sleep(self._retry_delay_seconds)

        # This line should never be reached.
        raise AssertionError("Unreachable code.")
