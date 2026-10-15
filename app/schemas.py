"""Pydantic models for API responses."""

from datetime import datetime, timezone

from pydantic import BaseModel, field_validator


class Segment(BaseModel):
    start: float
    end: float
    text: str


class JobCreatedResponse(BaseModel):
    job_id: str
    status: str


class JobResponse(BaseModel):
    job_id: str
    status: str
    filename: str
    attempts: int
    created_at: datetime
    completed_at: datetime | None = None
    language: str | None = None
    text: str | None = None
    segments: list[Segment] | None = None
    error: str | None = None
    retryable: bool | None = None

    @field_validator("created_at", "completed_at")
    @classmethod
    def assume_utc(cls, value: datetime | None) -> datetime | None:
        # SQLite drops timezone info; every timestamp we store is UTC.
        if value is not None and value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value


class HealthResponse(BaseModel):
    status: str
    database: str
    broker: str
