from fastapi import APIRouter

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
async def health() -> dict[str, str]:
    """
    Returns the health status of the application.
    """

    return {
        "status": "healthy",
    }
