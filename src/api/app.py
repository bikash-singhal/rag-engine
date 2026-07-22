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
    lifespan=lifespan,
)

app.include_router(chat_router)
app.include_router(ingest_router)
app.include_router(health_router)
