"""Celery application. Redis is the broker; results live in our own database."""

import logging

from celery import Celery
from celery.signals import worker_init, worker_process_init

from app.core.config import settings
from app.db.database import init_db

celery_app = Celery(
    "transcription",
    broker=settings.redis_url,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    # Job state and results are stored in the database, so no result backend.
    task_ignore_result=True,
    worker_concurrency=settings.celery_worker_concurrency,
    # Acknowledge a message only after the task finishes, so a crashed worker's
    # job is redelivered instead of lost. This is why tasks must be idempotent.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # Transcriptions are long; don't let one process reserve jobs it can't start.
    worker_prefetch_multiplier=1,
    # Redis redelivers unacked messages after this timeout. It must be longer
    # than the longest transcription plus retry countdown, or jobs run twice.
    broker_transport_options={"visibility_timeout": 4 * 3600},
    broker_connection_retry_on_startup=True,
)


@worker_init.connect
def _on_worker_init(**_kwargs) -> None:
    """Runs once in the main worker process, before child processes are forked."""
    init_db()


@worker_process_init.connect
def _on_worker_process_init(**_kwargs) -> None:
    """Runs once in each worker child process after it is forked."""
    logging.basicConfig(level=settings.log_level)
    # Load the model here, after the fork, so each process gets its own copy
    # up front instead of on the first job.
    from app.services.transcription import get_model

    try:
        get_model()
    except Exception:  # The task will try again and record a proper error.
        logging.getLogger(__name__).exception("Could not preload Whisper model")
