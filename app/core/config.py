"""Application configuration, loaded from environment variables (and an optional .env file)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Storage
    database_url: str = "sqlite:///./transcriptions.db"
    upload_dir: Path = Path("./uploads")
    max_upload_size_mb: int = 25
    delete_audio_after_success: bool = True

    # Celery / Redis
    redis_url: str = "redis://localhost:6379/0"
    celery_worker_concurrency: int = 2

    # Retries (exponential backoff: base * 2^retry, capped at max)
    task_max_retries: int = 3
    task_retry_backoff_base_seconds: int = 5
    task_retry_backoff_max_seconds: int = 300
    # A job stuck in "processing" longer than this is treated as abandoned
    # (e.g. the worker crashed) and may be picked up again.
    stale_job_seconds: int = 3600

    # Whisper
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_language: str | None = None  # None = auto-detect

    # Dev only: make the first N attempts of every job fail with a temporary
    # error, to demonstrate retries. Keep at 0 in real use.
    simulate_transient_failures: int = 0

    log_level: str = "INFO"

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
