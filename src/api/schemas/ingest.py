from pydantic import BaseModel, Field

from src.core.models import JobStatus


class IngestResponse(BaseModel):
    """
    Response returned after submitting
    an ingestion job.
    """

    status: str = Field(
        ...,
        examples=["accepted"],
    )

    document: str = Field(
        ...,
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

    job_id: str

    filename: str

    status: JobStatus

    error: str | None = None
