from pydantic import BaseModel, Field

from src.core.models import JobStatus


class IngestResponse(BaseModel):
    """
    Response returned after submitting
    an ingestion job.
    """

    status: str = Field(
        ...,
        description="Status of the ingestion request.",
        examples=["accepted"],
    )

    document: str = Field(
        ...,
        description="Name of the document submitted for ingestion.",
        examples=["annual_report.pdf"],
    )

    job_id: str = Field(
        ...,
        description="Identifier used to track ingestion progress.",
        examples=["8d5d61fd-4ef5-4bb8-a72e-b0cb98d5d12b"],
    )


class JobStatusResponse(BaseModel):
    """
    Status of an ingestion job.
    """

    job_id: str = Field(
        ...,
        description="Unique identifier of the ingestion job.",
    )

    filename: str = Field(
        ...,
        description="Name of the document being processed.",
    )

    status: JobStatus = Field(
        ...,
        description="Current state of the ingestion job.",
    )

    error: str | None = Field(
        default=None,
        description="Error message if the ingestion job failed.",
    )
