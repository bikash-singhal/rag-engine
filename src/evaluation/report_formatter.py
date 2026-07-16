from src.core.models import BenchmarkEvaluation, ExperimentReport


class ReportFormatter:
    """
    Formats evaluation reports for console output.
    """

    @staticmethod
    def print_benchmark_report(
        report: BenchmarkEvaluation,
    ) -> None:

        print()

        print("=" * 80)
        print("Retrieval Benchmark")
        print("=" * 80)

        print(f"Hit Rate : {report.average_hit_rate:.2f}")
        print(f"Precision: {report.average_precision:.2f}")
        print(f"Recall   : {report.average_recall:.2f}")
        print(f"F1 Score : {report.average_f1_score:.2f}")
        print(f"MRR      : {report.average_mrr:.2f}")

        print("=" * 80)

    @staticmethod
    def print_experiment_report(
        report: ExperimentReport,
    ) -> None:

        print()

        print("=" * 80)

        print(
            f"{'Retriever':<15}"
            f"{'Hit':>8}"
            f"{'Prec':>10}"
            f"{'Recall':>10}"
            f"{'F1':>10}"
            f"{'MRR':>10}"
        )

        print("=" * 80)

        for experiment in report.experiments:

            benchmark = experiment.benchmark

            print(
                f"{experiment.name:<15}"
                f"{benchmark.average_hit_rate:>8.2f}"
                f"{benchmark.average_precision:>10.2f}"
                f"{benchmark.average_recall:>10.2f}"
                f"{benchmark.average_f1_score:>10.2f}"
                f"{benchmark.average_mrr:>10.2f}"
            )

        print("=" * 80)
