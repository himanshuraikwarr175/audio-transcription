"""Worker task tests. transcribe_file is mocked; Celery runs the task in-process
with .apply(), which also runs retries immediately (countdowns are skipped)."""

from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from app.core.config import settings
from app.db.models import JobStatus, TranscriptionJob, utcnow
from app.services.transcription import InvalidAudioError
from app.workers.tasks import TASK_NAME, backoff_seconds, transcribe_audio
from tests.conftest import WAV_BYTES

RESULT = {
    "text": "Hello, how are you? I am doing well.",
    "language": "en",
    "segments": [
        {"start": 0.0, "end": 2.5, "text": "Hello, how are you?"},
        {"start": 2.5, "end": 5.0, "text": "I am doing well."},
    ],
}


@pytest.fixture
def job(db):
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    path = settings.upload_dir / "job-under-test.wav"
    path.write_bytes(WAV_BYTES)
    job = TranscriptionJob(
        original_filename="hello.wav",
        content_type="audio/wav",
        file_size=len(WAV_BYTES),
        file_path=str(path),
    )
    db.add(job)
    db.commit()
    yield job
    path.unlink(missing_ok=True)


def run(job_id):
    transcribe_audio.apply(args=[job_id])


def reload(db, job):
    db.expire_all()
    return db.get(TranscriptionJob, job.id)


def test_task_name_is_stable():
    assert transcribe_audio.name == TASK_NAME == "transcription.transcribe_audio"


def test_successful_transcription_saves_segments_and_deletes_audio(db, job):
    with patch("app.workers.tasks.transcribe_file", return_value=RESULT) as transcribe:
        run(job.id)

    transcribe.assert_called_once_with(job.file_path)
    job = reload(db, job)
    assert job.status == JobStatus.COMPLETED.value
    assert job.segments == RESULT["segments"]
    assert job.language == "en"
    assert job.attempts == 1
    assert not Path(job.file_path).exists()


def test_invalid_audio_fails_permanently_without_retry(db, job):
    with patch("app.workers.tasks.transcribe_file", side_effect=InvalidAudioError("Could not decode audio")) as t:
        run(job.id)

    assert t.call_count == 1
    job = reload(db, job)
    assert job.status == JobStatus.FAILED.value
    assert job.is_retryable is False
    assert "Could not decode audio" in job.error_message
    assert Path(job.file_path).exists()  # kept for investigation


def test_transient_error_is_retried_then_succeeds(db, job):
    side_effects = [RuntimeError("out of memory"), RuntimeError("out of memory"), RESULT]
    with patch("app.workers.tasks.transcribe_file", side_effect=side_effects) as t:
        run(job.id)

    assert t.call_count == 3
    job = reload(db, job)
    assert job.status == JobStatus.COMPLETED.value
    assert job.attempts == 3


def test_gives_up_after_max_retries(db, job):
    with patch("app.workers.tasks.transcribe_file", side_effect=RuntimeError("model crashed")) as t:
        run(job.id)

    assert t.call_count == settings.task_max_retries + 1
    job = reload(db, job)
    assert job.status == JobStatus.FAILED.value
    assert job.is_retryable is True
    assert "after 4 attempts" in job.error_message
    assert Path(job.file_path).exists()  # kept so the job can be retried manually


def test_completed_job_is_not_processed_again(db, job):
    with patch("app.workers.tasks.transcribe_file", return_value=RESULT) as t:
        run(job.id)
        run(job.id)  # e.g. a duplicate message delivery
    assert t.call_count == 1


def test_job_being_processed_is_not_claimed_twice(db, job):
    job.status = JobStatus.PROCESSING.value
    job.started_at = utcnow()
    db.commit()

    with patch("app.workers.tasks.transcribe_file", return_value=RESULT) as t:
        run(job.id)
    t.assert_not_called()


def test_stale_processing_job_is_reclaimed(db, job):
    job.status = JobStatus.PROCESSING.value
    job.started_at = utcnow() - timedelta(seconds=settings.stale_job_seconds + 60)
    db.commit()

    with patch("app.workers.tasks.transcribe_file", return_value=RESULT) as t:
        run(job.id)
    t.assert_called_once()
    assert reload(db, job).status == JobStatus.COMPLETED.value


def test_unknown_job_is_ignored():
    with patch("app.workers.tasks.transcribe_file") as t:
        run("does-not-exist")
    t.assert_not_called()


def test_simulated_failures_trigger_retries(db, job):
    with (
        patch.object(settings, "simulate_transient_failures", 2),
        patch("app.workers.tasks.transcribe_file", return_value=RESULT) as t,
    ):
        run(job.id)
    assert t.call_count == 1
    job = reload(db, job)
    assert job.status == JobStatus.COMPLETED.value
    assert job.attempts == 3


def test_backoff_grows_exponentially_and_is_capped():
    with (
        patch.object(settings, "task_retry_backoff_base_seconds", 5),
        patch.object(settings, "task_retry_backoff_max_seconds", 60),
    ):
        assert 5 <= backoff_seconds(0) <= 5.5
        assert 10 <= backoff_seconds(1) <= 11
        assert 20 <= backoff_seconds(2) <= 22
        assert 60 <= backoff_seconds(10) <= 66
