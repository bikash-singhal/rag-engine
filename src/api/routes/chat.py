from fastapi import APIRouter, Depends

from src.api.dependencies import get_rag_engine
from src.app.rag_engine import RAGEngine

router = APIRouter()


@router.post("/")
def chat(
    engine: RAGEngine = Depends(get_rag_engine),
):

    return {
        "message": "RAG Engine injected successfully!",
        "engine_id": id(engine),
    }
