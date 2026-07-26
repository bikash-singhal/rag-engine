from collections.abc import Iterator
from pathlib import Path

from pypdf import PdfReader

from src.core.models import Page


def reader(
    pdf_path: str | Path,
) -> Iterator[Page]:
    """
    Reads a PDF file lazily.

    Args:
        pdf_path: Path to the PDF file.

    Yields:
        Page objects extracted from the PDF.
    """

    pdf_path = Path(pdf_path)

    try:
        pdf_reader = PdfReader(pdf_path)
    except Exception as exc:
        raise RuntimeError(f"Could not read PDF: {pdf_path}") from exc

    for page_number, page in enumerate(pdf_reader.pages, start=1):

        text = page.extract_text()

        if text and text.strip():

            yield Page(
                page_number=page_number,
                text=text,
                metadata={
                    "source": pdf_path.name,
                },
            )
