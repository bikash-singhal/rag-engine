from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    Incoming request for the chat endpoint.
    """

    query: str = Field(
        ...,
        description="User question.",
        examples=["What is Amazon SageMaker?"],
    )


class SourceResponse(BaseModel):
    """
    One supporting source returned with the answer.
    """

    document: str
    page: int


class ChatResponse(BaseModel):
    """
    Response returned by the chat endpoint.
    """

    answer: str
    sources: list[SourceResponse]
