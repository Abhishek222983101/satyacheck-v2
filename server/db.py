"""SQLite persistence per blueprint §6.2. SQLModel, WAL mode, one file.
Data dir: SATYA_VAR env (default ./var), resolved to absolute at import so worker
threads and tests can chdir freely."""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlmodel import Field, SQLModel, Session, create_engine, select

VAR = Path(os.environ.get("SATYA_VAR", "var")).resolve()
DB_PATH = VAR / "satya.db"
TTL_HOURS = 24

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False, "timeout": 30})


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Case(SQLModel, table=True):
    id: str = Field(primary_key=True)
    created_at: datetime = Field(default_factory=_now)
    source: str  # upload | sample | webcam | redteam | chain
    title: str | None = None
    expires_at: datetime


class Media(SQLModel, table=True):
    id: str = Field(primary_key=True)
    case_id: str = Field(foreign_key="case.id", index=True)
    kind: str  # original | gen | attack | resync
    gen: int | None = Field(default=None, ge=0, le=5)
    parent_media_id: str | None = Field(default=None, foreign_key="media.id")
    path: str
    sha256: str
    duration_s: float | None = None
    width: int | None = None
    height: int | None = None
    has_audio: bool = False
    emulator_params: str | None = None  # JSON (ForwardChain draw) when kind='gen'


class Analysis(SQLModel, table=True):
    id: str = Field(primary_key=True)
    media_id: str = Field(foreign_key="media.id", index=True)
    status: str = "queued"  # queued | running | done | error
    label: str | None = None  # REAL | SUSPECT | FAKE
    p_fake: float | None = None
    reason: str | None = None
    bucket: str | None = None
    result_json: str | None = None  # full AnalysisResult v2
    error: str | None = None
    total_ms: int | None = None
    model_versions: str = "{}"
    created_at: datetime = Field(default_factory=_now)
    finished_at: datetime | None = None


class Artifact(SQLModel, table=True):
    id: str = Field(primary_key=True)
    analysis_id: str = Field(foreign_key="analysis.id", index=True)
    name: str
    path: str
    mime: str


class Chain(SQLModel, table=True):
    id: str = Field(primary_key=True)
    case_id: str = Field(foreign_key="case.id")
    source_media_id: str = Field(foreign_key="media.id")
    status: str = "running"
    created_at: datetime = Field(default_factory=_now)


class Attack(SQLModel, table=True):
    id: str = Field(primary_key=True)
    case_id: str = Field(foreign_key="case.id")
    kind: str  # splice_delete | splice_swap | tts_insert | lipsync_*
    params: str  # JSON
    input_media_id: str = Field(foreign_key="media.id")
    output_media_id: str | None = Field(default=None, foreign_key="media.id")
    status: str = "queued"


def init_db() -> None:
    VAR.mkdir(exist_ok=True)
    SQLModel.metadata.create_all(engine)
    janitor()


def janitor() -> int:
    """Delete cases past expiry (cascades to media/analyses/artifacts rows; files swept separately)."""
    import shutil
    n = 0
    with Session(engine) as s:
        expired = s.exec(select(Case).where(Case.expires_at < _now())).all()
        for c in expired:
            shutil.rmtree(VAR / "uploads" / c.id, ignore_errors=True)
            shutil.rmtree(VAR / "out" / c.id, ignore_errors=True)
            s.delete(c)
            n += 1
        s.commit()
    return n


def new_case(source: str, title: str | None = None) -> Case:
    import uuid
    c = Case(id=str(uuid.uuid4()), source=source, title=title, expires_at=_now() + timedelta(hours=TTL_HOURS))
    return c


def j(obj) -> str:
    return json.dumps(obj, default=float)
