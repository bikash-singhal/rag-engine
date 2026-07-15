from collections.abc import Iterator

from src.core.models import Page


class Preprocessor:
    """
    Cleans document pages before chunking.
    """

    def preprocess(
        self,
        pages: Iterator[Page],
    ) -> Iterator[Page]:
        """
        Cleans pages while preserving streaming.
        """

        for page in pages:

            text = self._clean_text(
                page.text
            )

            if not text:
                continue

            yield Page(
                page_number=page.page_number,
                text=text,
                metadata=page.metadata,
            )

    @staticmethod
    def _clean_text(
        text: str,
    ) -> str:
        """
        Performs lightweight text normalization.
        """

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        return "\n".join(lines).strip()