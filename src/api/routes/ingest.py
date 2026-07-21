from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile

from src.api.dependencies import get_job_manager, get_rag_engine
from src.api.job_manager import JobManager
from src.api.schemas.ingest import IngestResponse, JobStatusResponse
from src.app.rag_engine import RAGEngine
from src.core.models import JobStatus

router = APIRouter(
    prefix="/ingest",
    tags=["Ingest"],
)


@router.post(
    "",
    response_model=IngestResponse,
)
async def ingest(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    engine: RAGEngine = Depends(get_rag_engine),
    job_manager: JobManager = Depends(get_job_manager),
) -> IngestResponse:

    temp_path = save_upload_to_temp(file)

    job = job_manager.create_job(
        file.filename or "upload.pdf",
    )

    background_tasks.add_task(
        ingest_uploaded_file,
        engine,
        job_manager,
        job.id,
        temp_path,
    )

    return IngestResponse(
        status="accepted",
        document=file.filename or "upload.pdf",
        job_id=job.id,
    )


def save_upload_to_temp(
    file: UploadFile,
) -> Path:
    """
    Save uploaded PDF to a temporary file.
    """

    suffix = Path(file.filename or "upload.pdf").suffix

    with NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp_file:

        data = file.file.read()

        temp_file.write(data)

        return Path(temp_file.name)


def ingest_uploaded_file(
    engine: RAGEngine,
    job_manager: JobManager,
    job_id: str,
    temp_path: Path,
) -> None:
    """
    Background task that ingests a document
    and updates job status.
    """

    job_manager.update_status(
        job_id,
        JobStatus.RUNNING,
    )

    try:

        engine.ingest(temp_path)

        job_manager.update_status(
            job_id,
            JobStatus.COMPLETED,
        )

    except Exception as ex:

        job_manager.mark_failed(
            job_id,
            str(ex),
        )

        raise

    finally:

        temp_path.unlink(
            missing_ok=True,
        )


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
)
async def get_job_status(
    job_id: str,
    job_manager: JobManager = Depends(get_job_manager),
) -> JobStatusResponse:

    job = job_manager.get_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found.",
        )

    return JobStatusResponse(
        job_id=job.id,
        filename=job.filename,
        status=job.status,
        error=job.error,
    )
