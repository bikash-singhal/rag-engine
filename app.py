from dotenv import load_dotenv

from src.app.rag_engine import RAGEngine


def main():

    load_dotenv()

    engine = RAGEngine()

    engine.load_or_ingest(
        document_directory="data/raw",
        index_directory="data/indexes/default",
    )

    while True:

        question = input("\nQuestion (type 'exit' to quit): ")

        if question.lower() == "exit":
            break

        print("\nSearching...\n")

        results = engine.retrieve(question)

        print("Retrieved Chunks\n")

        for i, result in enumerate(results, start=1):

            print(f"Result {i}")
            print(f"Score : {result.score:.4f}")
            print(f"Page  : {result.chunk.page_number}")
            print(result.chunk.text[:300])
            print("-" * 80)

        print("\nGenerating answer...\n")

        print(engine.ask(question))


if __name__ == "__main__":
    main()
