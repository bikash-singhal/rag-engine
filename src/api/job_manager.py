from threading import Lock

from src.core.models import IngestJob, JobStatus


class JobManager:
    """
    In-memory registry for ingestion jobs.
    """

    def __init__(self) -> None:

        self._jobs: dict[str, IngestJob] = {}
        self._lock = Lock()

    def create_job(
        self,
        filename: str,
    ) -> IngestJob:

        with self._lock:

            job = IngestJob(filename)

            self._jobs[job.id] = job

            return job

    def get_job(
        self,
        job_id: str,
    ) -> IngestJob | None:

        with self._lock:
            return self._jobs.get(job_id)

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
    ) -> None:

        with self._lock:

            job = self._jobs.get(job_id)

            if job is not None:
                job.status = status

    def mark_failed(
        self,
        job_id: str,
        error: str,
    ) -> None:

        with self._lock:

            job = self._jobs.get(job_id)

            if job is not None:
                job.status = JobStatus.FAILED
                job.error = error
