from src.core.models import BenchmarkSummary


class ReportFormatter:

    @staticmethod
    def format(
        summary: BenchmarkSummary,
    ) -> str:

        lines = []

        lines.append("=" * 50)
        lines.append("Benchmark Summary")
        lines.append("=" * 50)
        lines.append("")

        lines.append(f"Benchmark : {summary.benchmark_name}")
        lines.append(f"Cases     : {summary.total_cases}")
        lines.append(f"Passed    : {summary.passed_cases}")
        lines.append(f"Failed    : {summary.failed_cases}")
        lines.append("")

        lines.append("Metrics")
        lines.append("")

        for metric, score in summary.metric_scores.items():

            lines.append(f"  {metric:<10}: {score:.3f}")

        return "\n".join(lines)
