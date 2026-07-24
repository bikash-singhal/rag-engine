from src.config.settings import RETRIEVAL_TOP_K
from src.core.models import BenchmarkDataset, BenchmarkSummary
from src.evaluation.generation_evaluator import GenerationEvaluator
from src.evaluation.retrieval_evaluator import RetrievalEvaluator


class BenchmarkRunner:

    def __init__(
        self,
        chat_engine,
        retrieval_evaluator: RetrievalEvaluator,
        generation_evaluator: GenerationEvaluator | None = None,
    ):
        self.chat_engine = chat_engine
        self.retrieval_evaluator = retrieval_evaluator
        self.generation_evaluator = generation_evaluator

    def run(
        self,
        dataset: BenchmarkDataset,
        experiment_name: str,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> BenchmarkSummary:

        results = []

        for index, case in enumerate(dataset.cases, start=1):

            chat_result = self.chat_engine.evaluate(case.question)

            retrieved = chat_result.retrieved_chunks

            result = self.retrieval_evaluator.evaluate(
                case,
                retrieved,
            )

            if self.generation_evaluator:
                try:
                    generation_scores = self.generation_evaluator.evaluate(
                        question=case.question,
                        contexts=[result.chunk.text for result in retrieved],
                        answer=chat_result.answer,
                        expected_answer=getattr(case, "expected_answer", None),
                    )
                except Exception:
                    generation_scores = {}

                result.metric_scores.update(generation_scores)

                print("-" * 80)
                print(f"Case {index}/{len(dataset.cases)}")
                print(f"Question : {case.question}")
                print()

                print(
                    f"Recall={result.metric_scores['Recall@K']:.3f} | "
                    f"MRR={result.metric_scores['MRR']:.3f} | "
                    f"Faith={result.metric_scores['faithfulness']:.3f}"
                )

                notes = []

                if result.metric_scores["MRR"] < 1.0:
                    notes.append("Retrieval")

                if result.metric_scores["faithfulness"] < 1.0:
                    notes.append("Faithfulness")

                if notes:
                    print(f"Observations: {', '.join(notes)}")

            results.append(result)

        if not results:
            raise ValueError("Benchmark dataset contains no test cases.")

        passed = sum(r.passed for r in results)

        metric_scores = {}

        all_metric_names = results[0].metric_scores.keys()

        for metric_name in all_metric_names:

            metric_scores[metric_name] = sum(
                result.metric_scores[metric_name] for result in results
            ) / len(results)

        return BenchmarkSummary(
            experiment_name=experiment_name,
            benchmark_name=dataset.name,
            total_cases=len(results),
            metric_scores=metric_scores,
            passed_cases=passed,
            failed_cases=len(results) - passed,
            results=results,
        )
