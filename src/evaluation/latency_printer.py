import logging

from src.core.latency import LatencyReport

logger = logging.getLogger(__name__)


class LatencyPrinter:

    @staticmethod
    def print(report: LatencyReport) -> None:
        logger.info("=" * 50)
        logger.info("Latency Report")
        logger.info("=" * 50)

        for stage, elapsed_ms in report.stages.items():
            logger.info(
                "%-22s : %8.2f ms",
                stage,
                elapsed_ms,
            )

        logger.info("-" * 50)
        logger.info(
            "%-22s : %8.2f ms",
            "TOTAL",
            report.total_ms,
        )
        logger.info("=" * 50)
