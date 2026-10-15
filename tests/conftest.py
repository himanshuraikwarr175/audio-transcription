"""Test setup: isolated SQLite DB and upload dir, no Redis, no Whisper model."""

import os
import shutil
import tempfile

# Must be set before any `app` module is imported, because settings are read at import time.
_tmp_dir = tempfile.mkdtemp(prefix="transcription-tests-")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp_dir}/test.db"
os.environ["UPLOAD_DIR"] = f"{_tmp_dir}/uploads"
os.environ["REDIS_URL"] = "redis://localhost:1/0"  # nothing listens here
os.environ["MAX_UPLOAD_SIZE_MB"] = "1"
os.environ["TASK_RETRY_BACKOFF_BASE_SECONDS"] = "0"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.db.database import Base, SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402

# Small but valid file headers, enough to pass the API's magic-byte check.
WAV_BYTES = b"RIFF" + b"\x24\x00\x00\x00" + b"WAVEfmt " + b"\x00" * 64
MP3_BYTES = b"ID3" + b"\x00" * 64
M4A_BYTES = b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 64


@pytest.fixture(autouse=True)
def clean_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def db():
    session = SessionLocal()
    yield session
    session.close()


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_tmp_dir, ignore_errors=True)
