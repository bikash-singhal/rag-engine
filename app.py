from dotenv import load_dotenv

from src.app.rag_engine import RAGEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():

    load_dotenv()

    engine = RAGEngine()

    engine.load_or_ingest(
        document_directory="data/documents",
        index_directory="data/indexes/default",
    )

    while True:

        question = input("\nQuestion (type 'exit' to quit): ").strip()
        if not question:
            continue

        logger.info("User question received.")

        if question.lower() == "exit":
            logger.info("Exiting application.")
            break

        logger.info("Searching...")

        results = engine.retrieve(question)

        logger.info("Retrieved %d search results", len(results))

        for i, result in enumerate(results, start=1):

            logger.debug("Result %d", i)
            logger.debug("Score: %.4f", result.score)
            logger.debug("Page: %d", result.chunk.page_number)
            logger.debug("Chunk Index: %d", result.chunk.chunk_index)
            logger.debug(result.chunk.text[:300])
            logger.debug("-" * 80)

        logger.info("Generating answer...")

        for token in engine.ask(question):
            print(token, end="", flush=True)

        print()


if __name__ == "__main__":
    main()
