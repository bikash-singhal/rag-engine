from collections.abc import Callable, Iterator
from pathlib import Path

from src.core.models import Page
from src.embeddings.embedder import Embedder
from src.ingestion.chunker import Chunker
from src.ingestion.preprocessor import Preprocessor
from src.utils.logger import get_logger
from src.vectorstores.faiss_store import FAISSVectorStore

logger = get_logger(__name__)


class DocumentIndexer:
    """
    Orchestrates the complete document indexing pipeline.

    Pipeline:
        Reader
          ↓
        Preprocessor
          ↓
        Chunker
          ↓
        Embedder
          ↓
        Vector Store
    """

    def __init__(
        self,
        reader: Callable[[str | Path], Iterator[Page]],
        preprocessor: Preprocessor,
        chunker: Chunker,
        embedder: Embedder,
        vector_store: FAISSVectorStore,
    ) -> None:
        self.reader = reader
        self.preprocessor = preprocessor
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    def index(
        self,
        file_path: str | Path,
    ) -> None:
        """
        Index a document into the vector store.

        Args:
            file_path: Path to the document.
        """

        document_pages = self.reader(file_path)

        document_pages = self.preprocessor.preprocess(document_pages)

        chunks = self.chunker.chunk(document_pages)

        embedded_chunks = self.embedder.embed(chunks)

        self.vector_store.add(
            embedded_chunks=embedded_chunks,
        )

    def index_directory(
        self,
        directory: str | Path,
    ) -> None:
        """
        Indexes all PDF files inside a directory.
        """

        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        pdf_files = sorted(directory.rglob("*.pdf"))

        if not pdf_files:
            return

        for pdf_file in pdf_files:

            try:

                self.index(pdf_file)

            except Exception as exc:
                logger.exception(
                    "Failed to index %s",
                    pdf_file,
                )
