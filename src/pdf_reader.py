from pathlib import Path

from pypdf import PdfReader

from src.models import Document


def read_pdf(
    pdf_path: str | Path,
) -> Document:
    """
    Reads a PDF file and extracts all readable text.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A Document object containing the extracted text.
    """

    pdf_path = Path(pdf_path)

    try:
        reader = PdfReader(pdf_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not read PDF: {pdf_path}"
        ) from exc

    pages: list[str] = []

    for page in reader.pages:

        text = page.extract_text()

        if text and text.strip():
            pages.append(text.strip())

    full_text = "\n\n".join(pages)

    if not full_text:
        raise ValueError(
            f"No readable text found in '{pdf_path.name}'."
        )

    return Document(
        source=pdf_path.name,
        text=full_text,
    )