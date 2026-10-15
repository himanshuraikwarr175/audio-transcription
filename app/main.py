"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import settings
from app.db.database import init_db

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    init_db()
    yield


app = FastAPI(
    title="Audio Transcription Service",
    description="Upload audio, get back Whisper transcriptions with per-segment timestamps.",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(router)
