from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes.chat import router as chat_router
from src.api.routes.health import router as health_router
from src.api.routes.ingest import router as ingest_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once when the application starts
    and once when it shuts down.
    """

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

    ## Workflow

    1. Rewrite user query
    2. Generate multiple search queries
    3. Perform hybrid retrieval
    4. Rerank retrieved chunks
    5. Build context-aware prompt
    6. Generate response using AWS Bedrock
    7. Return answer with supporting sources
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
            "description": "Ingest documents into the knowledge base.",
        },
    ],
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(ingest_router)
