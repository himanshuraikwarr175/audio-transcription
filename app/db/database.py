"""Database engine and session setup."""

from collections.abc import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings

_is_sqlite = settings.database_url.startswith("sqlite")

engine = create_engine(
    settings.database_url,
    # SQLite connections are used from FastAPI's thread pool.
    connect_args={"check_same_thread": False} if _is_sqlite else {},
)

if _is_sqlite:

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_connection, _record) -> None:
        # The API and several worker processes write to the same file.
        # WAL lets readers and a writer work at the same time, and
        # busy_timeout makes a writer wait for the lock instead of failing.
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    # Import models so they are registered on Base.metadata.
    from app.db import models  # noqa: F401

    try:
        Base.metadata.create_all(bind=engine)
    except OperationalError as exc:
        # The API and the worker may start at the same moment on a fresh
        # database; if the other one created the table first, that's fine.
        if "already exists" not in str(exc):
            raise


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
