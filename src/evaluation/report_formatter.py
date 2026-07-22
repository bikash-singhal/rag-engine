from src.core.models import BenchmarkSummary


class ReportFormatter:

    @staticmethod
    def format(
        summary: BenchmarkSummary,
    ) -> str:

        lines = []

        lines.append("=" * 60)
        lines.append("BENCHMARK SUMMARY")
        lines.append("=" * 60)
        lines.append("")

        lines.append(f"Experiment : {summary.experiment_name}")
        lines.append(f"Benchmark  : {summary.benchmark_name}")
        lines.append("")

        lines.append("Results")
        lines.append("-" * 60)
        lines.append(f"Total Cases : {summary.total_cases}")
        lines.append(f"Passed      : {summary.passed_cases}")
        lines.append(f"Failed      : {summary.failed_cases}")
        lines.append(f"Success Rate: {summary.passed_cases / summary.total_cases:.1%}")
        lines.append("")

        lines.append("Metrics")
        lines.append("-" * 60)

        for metric, score in sorted(summary.metric_scores.items()):

            lines.append(f"{metric:<15}: {score:.3f}")

        return "\n".join(lines)
