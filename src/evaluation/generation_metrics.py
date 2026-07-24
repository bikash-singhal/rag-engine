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

    def __init__(self) -> None:
        self._metric = Faithfulness(
            llm=create_ragas_llm(),
        )

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

        return float(self._metric.single_turn_score(sample))
