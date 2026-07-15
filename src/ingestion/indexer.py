from pathlib import Path


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
        preprocessor,
        chunker,
        embedder,
        vector_store,
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
            print(f"No PDF files found in {directory}")
            return

        indexed = 0
        failed = 0

        print(f"\nFound {len(pdf_files)} PDF(s).\n")

        for pdf_file in pdf_files:

            try:

                print(f"Indexing: {pdf_file.name}")

                self.index(pdf_file)

                indexed += 1

            except Exception as exc:

                failed += 1

                print(f"Failed: {pdf_file.name}")

                print(exc)

        print()

        print(f"Indexed : {indexed}")

        print(f"Failed  : {failed}")
