"""Database models. Only metadata and results are stored here; audio stays on disk."""

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TranscriptionJob(Base):
    __tablename__ = "transcription_jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: uuid.uuid4().hex)
    status: Mapped[str] = mapped_column(String(20), default=JobStatus.PENDING.value, index=True)

    # Upload info
    original_filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str | None] = mapped_column(String(100))
    file_size: Mapped[int] = mapped_column(Integer)
    file_path: Mapped[str] = mapped_column(String(500))  # path on disk, not the audio itself

    # Results
    text: Mapped[str | None] = mapped_column(Text)
    segments: Mapped[list | None] = mapped_column(JSON)
    language: Mapped[str | None] = mapped_column(String(20))

    # Failure tracking
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text)
    is_retryable: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
