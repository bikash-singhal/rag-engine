from src.evaluation.generation_metrics import GenerationMetric


class GenerationEvaluator:
    def __init__(self, metrics: list[GenerationMetric]) -> None:
        self._metrics = metrics

    def evaluate(
        self,
        question: str,
        contexts: list[str],
        answer: str,
        expected_answer: str | None = None,
    ) -> dict[str, float]:

        results = {}

        for metric in self._metrics:
            results[metric.name] = metric.evaluate(
                question=question,
                contexts=contexts,
                answer=answer,
                expected_answer=expected_answer,
            )

        return results
