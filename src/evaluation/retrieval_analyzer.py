from src.core.models import RetrievalReport, RetrievalStatistics, SearchResult


class RetrievalAnalyzer:
    """
    Computes retrieval statistics and generates a retrieval report.
    """

    def analyze(
        self,
        question: str,
        results: list[SearchResult],
    ) -> RetrievalReport:

        if not results:

            statistics = RetrievalStatistics(
                result_count=0,
                highest_score=0.0,
                lowest_score=0.0,
                average_score=0.0,
                unique_pages=[],
                page_diversity=0.0,
            )

            return RetrievalReport(
                question=question,
                results=[],
                statistics=statistics,
            )

        scores = [result.score for result in results]

        pages = [result.chunk.page_number for result in results]

        unique_pages = sorted(set(pages))

        statistics = RetrievalStatistics(
            result_count=len(results),
            highest_score=max(scores),
            lowest_score=min(scores),
            average_score=sum(scores) / len(scores),
            unique_pages=unique_pages,
            page_diversity=len(unique_pages) / len(results),
        )

        return RetrievalReport(
            question=question,
            results=results,
            statistics=statistics,
        )
