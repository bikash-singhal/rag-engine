from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.api.dependencies import get_chat_engine
from src.api.schemas.chat import ChatRequest, ChatResponse, SourceResponse
from src.chat.chat_engine import ChatEngine

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


@router.post(
    "",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
    engine: ChatEngine = Depends(get_chat_engine),
) -> ChatResponse:
    """
    Answers a user question using the RAG pipeline.
    """

    result = engine.chat(
        request.question,
    )

    return ChatResponse(
        answer=result.answer,
        rewritten_question=result.rewritten_question,
        sources=[
            SourceResponse(
                document=source.chunk.source,
                page=source.chunk.page_number,
                score=source.score,
            )
            for source in result.retrieved_chunks
        ],
        latency=result.latency,
    )


@router.post(
    "/stream",
)
def chat_stream(
    request: ChatRequest,
    engine: ChatEngine = Depends(get_chat_engine),
):
    """
    Streams the LLM response token-by-token.
    """

    return StreamingResponse(
        engine.ask(request.question),
        media_type="text/event-stream",
    )
