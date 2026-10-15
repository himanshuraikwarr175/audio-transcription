"""API tests. The Celery task is mocked, so no Redis or model is needed."""

from pathlib import Path
from unittest.mock import patch

import pytest

from app.db.models import JobStatus, TranscriptionJob
from tests.conftest import M4A_BYTES, MP3_BYTES, WAV_BYTES


@pytest.fixture
def mock_delay():
    with patch("app.api.routes.transcribe_audio.delay") as delay:
        yield delay


def upload(client, name="audio.wav", data=WAV_BYTES, content_type="audio/wav"):
    return client.post("/transcriptions", files={"file": (name, data, content_type)})


def test_health_reports_components(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["database"] == "ok"
    assert body["broker"] == "unavailable"  # no Redis in tests
    assert body["status"] == "degraded"


@pytest.mark.parametrize(
    "name,data,content_type",
    [
        ("a.wav", WAV_BYTES, "audio/wav"),
        ("a.mp3", MP3_BYTES, "audio/mpeg"),
        ("a.m4a", M4A_BYTES, "audio/mp4"),
        ("a.MP3", MP3_BYTES, "application/octet-stream"),
    ],
)
def test_upload_accepts_supported_formats(client, db, mock_delay, name, data, content_type):
    response = upload(client, name, data, content_type)

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "pending"
    mock_delay.assert_called_once_with(body["job_id"])

    job = db.get(TranscriptionJob, body["job_id"])
    assert job.original_filename == name
    assert job.file_size == len(data)
    assert Path(job.file_path).read_bytes() == data


def test_concurrent_uploads_get_unique_files(client, db, mock_delay):
    ids = [upload(client, "same-name.wav").json()["job_id"] for _ in range(3)]
    assert len(set(ids)) == 3
    paths = {db.get(TranscriptionJob, job_id).file_path for job_id in ids}
    assert len(paths) == 3


def test_rejects_unsupported_extension(client, mock_delay):
    response = upload(client, "notes.txt", b"hello", "text/plain")
    assert response.status_code == 415
    assert "Unsupported file extension" in response.json()["detail"]
    mock_delay.assert_not_called()


def test_rejects_mismatched_content_type(client, mock_delay):
    response = upload(client, "audio.wav", WAV_BYTES, "image/png")
    assert response.status_code == 415


def test_rejects_file_that_is_not_really_audio(client, mock_delay):
    response = upload(client, "fake.mp3", b"this is just text", "audio/mpeg")
    assert response.status_code == 400
    assert "not a valid .mp3" in response.json()["detail"]
    mock_delay.assert_not_called()


def test_rejects_empty_file(client, mock_delay):
    response = upload(client, "empty.wav", b"", "audio/wav")
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_rejects_oversized_file(client, mock_delay):
    too_big = WAV_BYTES + b"\x00" * (1024 * 1024)  # limit is 1 MB in tests
    response = upload(client, "big.wav", too_big, "audio/wav")
    assert response.status_code == 413
    mock_delay.assert_not_called()


def test_missing_file_field(client):
    response = client.post("/transcriptions")
    assert response.status_code == 422


def test_upload_returns_503_when_queue_is_down(client, db):
    with patch("app.api.routes.transcribe_audio.delay", side_effect=ConnectionError("redis down")):
        response = upload(client)
    assert response.status_code == 503
    job_id = response.json()["detail"]["job_id"]
    job = db.get(TranscriptionJob, job_id)
    assert job.status == JobStatus.FAILED.value
    assert job.is_retryable


def test_get_unknown_job_returns_404(client):
    response = client.get("/transcriptions/doesnotexist")
    assert response.status_code == 404


def test_get_pending_job(client, mock_delay):
    job_id = upload(client).json()["job_id"]
    body = client.get(f"/transcriptions/{job_id}").json()
    assert body["status"] == "pending"
    assert body["segments"] is None


def test_get_completed_job_returns_segments(client, db, mock_delay):
    job_id = upload(client).json()["job_id"]
    job = db.get(TranscriptionJob, job_id)
    job.status = JobStatus.COMPLETED.value
    job.text = "Hello, how are you? I am doing well."
    job.segments = [
        {"start": 0.0, "end": 2.5, "text": "Hello, how are you?"},
        {"start": 2.5, "end": 5.0, "text": "I am doing well."},
    ]
    db.commit()

    body = client.get(f"/transcriptions/{job_id}").json()
    assert body["status"] == "completed"
    assert body["segments"][1] == {"start": 2.5, "end": 5.0, "text": "I am doing well."}
    assert body["error"] is None


def test_get_failed_job_returns_error(client, db, mock_delay):
    job_id = upload(client).json()["job_id"]
    job = db.get(TranscriptionJob, job_id)
    job.status = JobStatus.FAILED.value
    job.error_message = "Could not decode audio"
    job.is_retryable = False
    db.commit()

    body = client.get(f"/transcriptions/{job_id}").json()
    assert body["status"] == "failed"
    assert body["error"] == "Could not decode audio"
    assert body["retryable"] is False


def test_retry_failed_job(client, db, mock_delay):
    job_id = upload(client).json()["job_id"]
    job = db.get(TranscriptionJob, job_id)
    job.status = JobStatus.FAILED.value
    job.error_message = "Transcription failed after 4 attempts"
    db.commit()
    mock_delay.reset_mock()

    response = client.post(f"/transcriptions/{job_id}/retry")
    assert response.status_code == 202
    assert response.json() == {"job_id": job_id, "status": "pending"}
    mock_delay.assert_called_once_with(job_id)


def test_retry_rejects_non_failed_job(client, mock_delay):
    job_id = upload(client).json()["job_id"]
    response = client.post(f"/transcriptions/{job_id}/retry")
    assert response.status_code == 409


def test_retry_rejects_permanent_failure(client, db, mock_delay):
    job_id = upload(client).json()["job_id"]
    job = db.get(TranscriptionJob, job_id)
    job.status = JobStatus.FAILED.value
    job.is_retryable = False
    db.commit()

    response = client.post(f"/transcriptions/{job_id}/retry")
    assert response.status_code == 409
    assert "permanently" in response.json()["detail"]


def test_retry_rejects_when_audio_was_deleted(client, db, mock_delay):
    job_id = upload(client).json()["job_id"]
    job = db.get(TranscriptionJob, job_id)
    job.status = JobStatus.FAILED.value
    db.commit()
    Path(job.file_path).unlink()

    response = client.post(f"/transcriptions/{job_id}/retry")
    assert response.status_code == 409


def test_retry_unknown_job_returns_404(client):
    assert client.post("/transcriptions/nope/retry").status_code == 404
