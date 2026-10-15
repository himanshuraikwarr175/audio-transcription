"""HTTP endpoints. These only validate, store and enqueue; they never run the model."""

import logging
import uuid
from pathlib import Path

import redis
from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy import text, update
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.database import get_db
from app.db.models import JobStatus, TranscriptionJob
from app.schemas import HealthResponse, JobCreatedResponse, JobResponse
from app.workers.tasks import transcribe_audio

logger = logging.getLogger(__name__)
router = APIRouter()

CHUNK_SIZE = 1024 * 1024

ALLOWED_CONTENT_TYPES = {
    ".wav": {"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"},
    ".mp3": {"audio/mpeg", "audio/mp3"},
    ".m4a": {"audio/mp4", "audio/x-m4a", "audio/m4a"},
}
# Many clients (curl included) send this when they don't know the type.
# We accept it because the file's magic bytes are checked anyway.
GENERIC_CONTENT_TYPE = "application/octet-stream"


@router.get("/health", response_model=HealthResponse)
def health(db: Session = Depends(get_db)) -> HealthResponse:
    try:
        db.execute(text("SELECT 1"))
        database = "ok"
    except Exception:
        logger.exception("Health check: database unavailable")
        database = "unavailable"

    try:
        redis.Redis.from_url(settings.redis_url, socket_connect_timeout=1, socket_timeout=1).ping()
        broker = "ok"
    except Exception:
        broker = "unavailable"

    overall = "ok" if database == "ok" and broker == "ok" else "degraded"
    return HealthResponse(status=overall, database=database, broker=broker)


@router.post(
    "/transcriptions",
    response_model=JobCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_transcription(file: UploadFile, db: Session = Depends(get_db)) -> JobCreatedResponse:
    # This is a plain `def`, so FastAPI runs it in a thread pool: many uploads
    # can be saved at once without blocking the event loop.
    extension = validate_upload_metadata(file)

    job_id = uuid.uuid4().hex
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    # The job ID makes the name unique, so concurrent uploads of "audio.mp3" never collide.
    destination = settings.upload_dir / f"{job_id}{extension}"
    size = save_upload(file, destination, extension)

    job = TranscriptionJob(
        id=job_id,
        original_filename=file.filename,
        content_type=file.content_type,
        file_size=size,
        file_path=str(destination),
    )
    db.add(job)
    db.commit()

    enqueue_or_fail(db, job)
    logger.info("Job %s: accepted %r (%d bytes)", job_id, file.filename, size)
    return JobCreatedResponse(job_id=job.id, status=job.status)


@router.get("/transcriptions/{job_id}", response_model=JobResponse)
def get_transcription(job_id: str, db: Session = Depends(get_db)) -> JobResponse:
    job = get_job_or_404(db, job_id)
    return to_response(job)


@router.post(
    "/transcriptions/{job_id}/retry",
    response_model=JobCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def retry_transcription(job_id: str, db: Session = Depends(get_db)) -> JobCreatedResponse:
    job = get_job_or_404(db, job_id)

    if job.status != JobStatus.FAILED.value:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Only failed jobs can be retried; this job is '{job.status}'.")
    if not job.is_retryable:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"This job failed permanently and will not succeed on retry: {job.error_message}",
        )
    if not Path(job.file_path).exists():
        raise HTTPException(status.HTTP_409_CONFLICT, "The audio file is no longer available; please upload it again.")

    # Conditional update so two simultaneous retry calls can't both enqueue.
    result = db.execute(
        update(TranscriptionJob)
        .where(TranscriptionJob.id == job_id, TranscriptionJob.status == JobStatus.FAILED.value)
        .values(status=JobStatus.PENDING.value, error_message=None, completed_at=None)
    )
    db.commit()
    if result.rowcount == 0:
        raise HTTPException(status.HTTP_409_CONFLICT, "This job is already being retried.")

    db.refresh(job)
    enqueue_or_fail(db, job)
    logger.info("Job %s: manually retried", job_id)
    return JobCreatedResponse(job_id=job.id, status=job.status)


# --- helpers ---------------------------------------------------------------


def validate_upload_metadata(file: UploadFile) -> str:
    """Check filename extension and declared content type. Returns the extension."""
    if not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The uploaded file has no filename.")

    extension = Path(file.filename).suffix.lower()
    if extension not in ALLOWED_CONTENT_TYPES:
        allowed = ", ".join(sorted(ALLOWED_CONTENT_TYPES))
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported file extension '{extension or '(none)'}'. Allowed: {allowed}.",
        )

    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type != GENERIC_CONTENT_TYPE and content_type not in ALLOWED_CONTENT_TYPES[extension]:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Content type '{content_type or '(none)'}' does not match a {extension} file.",
        )
    return extension


def looks_like_audio(header: bytes, extension: str) -> bool:
    """Cheap magic-byte check so a renamed text file is rejected up front.

    FFmpeg in the worker is the final judge; this only catches obvious junk.
    """
    if extension == ".wav":
        return header[:4] == b"RIFF" and header[8:12] == b"WAVE"
    if extension == ".mp3":
        # Either an ID3 tag or an MPEG audio frame sync (11 set bits).
        return header[:3] == b"ID3" or (len(header) > 1 and header[0] == 0xFF and header[1] & 0xE0 == 0xE0)
    if extension == ".m4a":
        return header[4:8] == b"ftyp"
    return False


def save_upload(file: UploadFile, destination: Path, extension: str) -> int:
    """Stream the upload to disk in chunks, enforcing the size limit. Returns bytes written."""
    max_bytes = settings.max_upload_size_bytes
    size = 0
    try:
        with destination.open("wb") as out:
            while chunk := file.file.read(CHUNK_SIZE):
                if size == 0 and not looks_like_audio(chunk[:12], extension):
                    raise HTTPException(
                        status.HTTP_400_BAD_REQUEST,
                        f"File content is not a valid {extension} audio file.",
                    )
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        413,
                        f"File exceeds the {settings.max_upload_size_mb} MB limit.",
                    )
                out.write(chunk)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise

    if size == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The uploaded file is empty.")
    return size


def enqueue_or_fail(db: Session, job: TranscriptionJob) -> None:
    """Send the job to Celery. If Redis is down, record that and return 503."""
    try:
        transcribe_audio.delay(job.id)
    except Exception as exc:
        logger.exception("Job %s: could not enqueue", job.id)
        job.status = JobStatus.FAILED.value
        job.error_message = f"Could not queue the job: {exc}"
        job.is_retryable = True  # The file is saved; retry once the broker is back.
        db.commit()
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            {"message": "Task queue is unavailable. Retry this job later.", "job_id": job.id},
        ) from exc


def get_job_or_404(db: Session, job_id: str) -> TranscriptionJob:
    job = db.get(TranscriptionJob, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Job '{job_id}' not found.")
    return job


def to_response(job: TranscriptionJob) -> JobResponse:
    fields = {}
    if job.status == JobStatus.COMPLETED.value:
        fields = {"language": job.language, "text": job.text, "segments": job.segments}
    elif job.status == JobStatus.FAILED.value:
        fields = {"error": job.error_message, "retryable": job.is_retryable}
    elif job.error_message:
        # Pending after a failed attempt: show why it's being retried.
        fields = {"error": job.error_message}

    return JobResponse(
        job_id=job.id,
        status=job.status,
        filename=job.original_filename,
        attempts=job.attempts,
        created_at=job.created_at,
        completed_at=job.completed_at,
        **fields,
    )
