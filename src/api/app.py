from fastapi import FastAPI

from src.api.routes.chat import router as chat_router

# from src.api.routes.evaluate import router as evaluate_router
from src.api.routes.health import router as health_router

# from src.api.routes.ingest import router as ingest_router

app = FastAPI(
    title="RAG Engine",
    version="1.0.0",
)

app.include_router(
    health_router,
    prefix="/health",
    tags=["Health"],
)


app.include_router(
    chat_router,
    prefix="/chat",
    tags=["Chat"],
)

""" 
app.include_router(
    ingest_router,
    prefix="/ingest",
    tags=["Ingestion"],
)

app.include_router(
    evaluate_router,
    prefix="/evaluate",
    tags=["Evaluation"],
)
 """
