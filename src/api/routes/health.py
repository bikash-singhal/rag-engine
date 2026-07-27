from fastapi import APIRouter, Depends

from src.api.dependencies import get_rag_engine
from src.app.rag_engine import RAGEngine

router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


@router.get(
    "",
    summary="Health check",
    description="""
Checks whether the API is running and ready to accept requests.

This endpoint is intended for container orchestration, monitoring,
and deployment verification.
""",
    response_description="Application health status.",
)
async def health(
    engine: RAGEngine = Depends(get_rag_engine),
) -> dict[str, object]:
    """
    Returns the health status of the application.
    """

    return {
        "status": "healthy",
        "knowledge_base_ready": engine.is_ready,
        "default_knowledge_base": (
            "Amazon SageMaker User Guide" if engine.is_ready else None
        ),
    }
