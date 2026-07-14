from src.core.models import Page


class DocumentIndexer:
    """
    Orchestrates the complete document indexing pipeline.

    Pipeline:
        Reader
            ↓
        Chunker
            ↓
        Embedder
            ↓
        Vector Store
    """

    def __init__(
        self,
        reader,
        chunker,
        embedder,
        vector_store,
    ) -> None:
        self.reader = reader
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    def index(
        self,
        file_path: str,
    ) -> None:
        """
        Index a document into the vector store.

        Args:
            file_path: Path to the document.
        """

        document_pages = self.reader(file_path)

        chunks = self.chunker.chunk(document_pages)

        embedded_chunks = self.embedder.embed(chunks)

        self.vector_store.add(
            embedded_chunks=embedded_chunks,
        )

