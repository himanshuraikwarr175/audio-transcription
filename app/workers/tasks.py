"""Background transcription task."""

import logging
import random
from datetime import timedelta
from pathlib import Path

from sqlalchemy import and_, or_, update
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.database import SessionLocal
from app.db.models import JobStatus, TranscriptionJob, utcnow
from app.services.transcription import (
    InvalidAudioError,
    TransientTranscriptionError,
    transcribe_file,
)
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

TASK_NAME = "transcription.transcribe_audio"


@celery_app.task(bind=True, name=TASK_NAME, max_retries=settings.task_max_retries)
def transcribe_audio(self, job_id: str) -> None:
    db = SessionLocal()
    try:
        job = claim_job(db, job_id)
        if job is None:
            return

        logger.info("Job %s: attempt %d (celery retry %d)", job_id, job.attempts, self.request.retries)
        try:
            if self.request.retries < settings.simulate_transient_failures:
                raise TransientTranscriptionError("Simulated temporary failure")
            result = transcribe_file(job.file_path)

        except InvalidAudioError as exc:
            # Permanent: retrying the same bad file would fail the same way.
            logger.warning("Job %s: invalid audio: %s", job_id, exc)
            mark_failed(db, job, str(exc), retryable=False)
            return

        except Exception as exc:
            if self.request.retries < self.max_retries:
                countdown = backoff_seconds(self.request.retries)
                logger.warning("Job %s: temporary failure (%s), retrying in %ds", job_id, exc, countdown)
                job.status = JobStatus.PENDING.value
                job.error_message = f"Attempt {job.attempts} failed: {exc}. Retrying in {countdown}s."
                db.commit()
                raise self.retry(exc=exc, countdown=countdown)

            logger.error("Job %s: giving up after %d attempts: %s", job_id, job.attempts, exc)
            mark_failed(db, job, f"Transcription failed after {job.attempts} attempts: {exc}", retryable=True)
            return

        job.status = JobStatus.COMPLETED.value
        job.text = result["text"]
        job.segments = result["segments"]
        job.language = result["language"]
        job.error_message = None
        job.completed_at = utcnow()
        db.commit()
        logger.info("Job %s: completed with %d segments", job_id, len(result["segments"]))

        # Delete the audio only after the result is safely committed.
        if settings.delete_audio_after_success:
            Path(job.file_path).unlink(missing_ok=True)
    finally:
        db.close()


def claim_job(db: Session, job_id: str) -> TranscriptionJob | None:
    """Atomically move a job to "processing", or return None if it shouldn't run.

    The UPDATE ... WHERE status = 'pending' is a single statement, so if the same
    message is delivered twice, only one worker wins the claim; the other sees
    0 rows updated and skips. Completed jobs are never reprocessed. A job left in
    "processing" by a crashed worker can be reclaimed once it is stale.
    """
    now = utcnow()
    stale_before = now - timedelta(seconds=settings.stale_job_seconds)
    result = db.execute(
        update(TranscriptionJob)
        .where(
            TranscriptionJob.id == job_id,
            or_(
                TranscriptionJob.status == JobStatus.PENDING.value,
                and_(
                    TranscriptionJob.status == JobStatus.PROCESSING.value,
                    TranscriptionJob.started_at < stale_before,
                ),
            ),
        )
        .values(
            status=JobStatus.PROCESSING.value,
            started_at=now,
            attempts=TranscriptionJob.attempts + 1,
        )
    )
    db.commit()

    if result.rowcount == 0:
        job = db.get(TranscriptionJob, job_id)
        if job is None:
            logger.error("Job %s: not found, skipping", job_id)
        else:
            logger.info("Job %s: status is %r, nothing to do", job_id, job.status)
        return None
    return db.get(TranscriptionJob, job_id)


def mark_failed(db: Session, job: TranscriptionJob, message: str, retryable: bool) -> None:
    job.status = JobStatus.FAILED.value
    job.error_message = message
    job.is_retryable = retryable
    job.completed_at = utcnow()
    db.commit()


def backoff_seconds(retries: int) -> int:
    """Exponential backoff with a little jitter: ~5s, ~10s, ~20s, ... capped."""
    delay = settings.task_retry_backoff_base_seconds * (2**retries)
    delay = min(delay, settings.task_retry_backoff_max_seconds)
    return int(delay + random.uniform(0, delay * 0.1))
