from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.dependencies import get_rag_engine
from src.api.routes.chat import router as chat_router
from src.api.routes.health import router as health_router
from src.api.routes.ingest import router as ingest_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once when the application starts
    and once when it shuts down.
    """
    get_rag_engine()  # Force initialization at startup
    yield


app = FastAPI(
    title="RAG Engine API",
    description="""
        Production-ready Retrieval-Augmented Generation (RAG) API built with FastAPI and AWS Bedrock.

        ## Features

        - Hybrid Retrieval (BM25 + FAISS)
        - Query Rewriting
        - Multi-query Expansion
        - CrossEncoder Reranking
        - AWS Bedrock (Nova Lite)
        - Streaming Responses
        - Document Ingestion
        - RAG Evaluation

        ## Demo Knowledge Base

        This project ships with a **pre-built knowledge base** created from the **Amazon SageMaker User Guide**.

        You can start chatting immediately without ingesting any documents.

        ### Try asking:

        - What is Amazon SageMaker?
        - Who developed SageMaker?
        - What are the prerequisites for Amazon SageMaker?
        - What is the Lakehouse architecture?
        - What frameworks are supported for data processing?
        - What data sources can SageMaker connect to?

        ## Using Your Own Documents

        Use the **Ingestion** endpoint to ingest your own PDF and replace the default knowledge base.

        ## Workflow

        1. Rewrite the user query.
        2. Generate multiple search queries.
        3. Perform hybrid retrieval.
        4. Rerank retrieved chunks.
        5. Build a context-aware prompt.
        6. Generate a response using AWS Bedrock.
        7. Return the answer with supporting sources.
        """,
    openapi_tags=[
        {
            "name": "Health",
            "description": "Application health and readiness checks.",
        },
        {
            "name": "Chat",
            "description": "Interact with the RAG engine.",
        },
        {
            "name": "Ingestion",
            "description": "Build or replace the knowledge base from your own PDF documents.",
        },
    ],
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(ingest_router)
