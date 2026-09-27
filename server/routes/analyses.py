"""Core analysis endpoints per blueprint §6.3: upload, SSE stream, result, artifacts, resync."""
from __future__ import annotations

import asyncio
import json
import subprocess
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse
from sqlmodel import Session, select

from server import runner
from server.db import Analysis, Artifact, Case, Media, engine, j, new_case
from server.util import OUT, UPLOADS, check_id, safe_artifact, save_upload

router = APIRouter(prefix="/v1")


def _events_response(analysis_id: str):
    async def gen():
        q = runner.subscribe(analysis_id)
        try:
            for ev in runner.replay(analysis_id):  # already-stored events first
                yield ev
            from sqlmodel import Session
            with Session(engine) as s:
                a = s.get(Analysis, analysis_id)
                done = a is not None and a.status in ("done", "error")
            if done:
                yield {"event": "end", "data": "{}"}
                return
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=300)
                except asyncio.TimeoutError:
                    yield {"event": "error", "data": j({"type": "timeout", "title": "Still working…", "detail": "This analysis is taking unusually long. Refresh to reconnect."})}
                    return
                yield ev
                if ev["event"] in ("end", "error"):
                    return
        finally:
            runner.unsubscribe(analysis_id, q)

    return EventSourceResponse(gen())


def _launch(analysis_id: str, media_path: str):
    loop = asyncio.get_running_loop()

    def on_event(kind: str, data: dict):
        loop.call_soon_threadsafe(asyncio.create_task, runner.emit(analysis_id, kind, data))

    def work():
        try:
            runner.run_analysis_sync(analysis_id, media_path, on_event=on_event)
            loop.call_soon_threadsafe(asyncio.create_task, runner.emit(analysis_id, "end", {}))
        except Exception as e:
            with Session(engine) as s:
                a = s.get(Analysis, analysis_id)
                if a:
                    a.status, a.error = "error", f"{e.__class__.__name__}: {e}"
                    s.add(a); s.commit()
            loop.call_soon_threadsafe(asyncio.create_task, runner.emit(
                analysis_id, "error", {"type": e.__class__.__name__, "title": "Analysis failed",
                                       "detail": "Something went wrong on our side — sorry! Your clip is safe; please try again."}))

    return loop.run_in_executor(None, work)


@router.post("/analyses", status_code=202)
async def create_analysis(file: UploadFile, background: BackgroundTasks, source: str = "upload", title: str | None = None):
    case = new_case(source if source in ("upload", "sample", "webcam", "redteam", "chain") else "upload", title)
    info = await save_upload(file, case.id, source)
    media_id = analysis_id = str(uuid.uuid4())
    case_id = case.id
    with Session(engine) as s:
        s.add(case); s.flush()
        s.add(Media(id=media_id, case_id=case_id, kind="original", path=info["path"], sha256=info["sha256"],
                    duration_s=info["duration_s"], width=info["width"], height=info["height"], has_audio=info["has_audio"]))
        s.flush()
        s.add(Analysis(id=analysis_id, media_id=media_id, status="queued", model_versions=j(runner.MODELS)))
        s.commit()
    _launch(analysis_id, info["path"])
    return {"id": analysis_id, "case_id": case_id, "events_url": f"/v1/analyses/{analysis_id}/events"}


@router.get("/analyses/{aid}/events")
async def events(aid: str):
    check_id(aid)
    return _events_response(aid)


@router.get("/analyses/{aid}")
async def get_analysis(aid: str):
    check_id(aid)
    with Session(engine) as s:
        a = s.get(Analysis, aid)
        if not a:
            from server.util import problem
            return problem(404, "Not found", "I can't find that analysis — it may have expired after 24 h.")
        if a.status != "done":
            return {"id": aid, "status": a.status, "error": a.error}
        return json.loads(a.result_json)


@router.get("/analyses/{aid}/artifacts/{name}")
async def artifact(aid: str, name: str):
    check_id(aid); safe_artifact(name)
    with Session(engine) as s:
        art = s.exec(select(Artifact).where(Artifact.analysis_id == aid, Artifact.name == name)).first()
    if not art:
        raise HTTPException(404, "no such artifact")
    return FileResponse(art.path, media_type=art.mime)


@router.post("/analyses/{aid}/resync")
async def resync(aid: str, body: dict):
    check_id(aid)
    off = float(body.get("offset_ms", 0))
    if not -2000 <= off <= 2000:
        raise HTTPException(422, "offset_ms must be within ±2000")
    with Session(engine) as s:
        m = s.get(Media, aid)
        if not m:
            raise HTTPException(404, "no such media")
        src, case_id = Path(m.path), m.case_id
    dst_dir = OUT / case_id
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"resync_{aid[:8]}_{int(off)}.mp4"
    if not dst.exists():
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-itsoffset", f"{off / 1000:.3f}",
                        "-i", str(src), "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", "-shortest", str(dst)], check=True)
        with Session(engine) as s:
            s.add(Artifact(id=str(uuid.uuid4()), analysis_id=aid, name=dst.name, path=str(dst), mime="video/mp4"))
            s.commit()
    return {"media_url": f"/v1/analyses/{aid}/artifacts/{dst.name}"}


@router.get("/analyses/{aid}/dossier.pdf")
async def dossier(aid: str):
    from server.dossier import render_dossier
    check_id(aid)
    return render_dossier(aid)


@router.get("/analyses/{aid}/card.png")
async def card(aid: str, lang: str = "en"):
    from server.dossier import render_card
    check_id(aid)
    return render_card(aid, lang)


@router.get("/analyses/{aid}/media")
async def media(aid: str):
    """The analysed clip itself (video element on the case board)."""
    check_id(aid)
    with Session(engine) as s:
        m = s.get(Media, aid)
    if not m:
        raise HTTPException(404, "no such media")
    return FileResponse(m.path, media_type="video/mp4")


@router.get("/analyses")
async def list_analyses(limit: int = 20):
    """Recent analyses for the Chain/Red-team pickers (safe subset only — no result JSON)."""
    limit = max(1, min(limit, 50))
    with Session(engine) as s:
        rows = s.exec(select(Analysis).order_by(Analysis.created_at.desc()).limit(limit)).all()
        out = []
        for a in rows:
            m = s.get(Media, a.media_id)
            name = m.path.split("/")[-1] if m else ""
            out.append({"analysis_id": a.id, "media_id": a.media_id, "status": a.status,
                        "label": a.label, "created_at": a.created_at.isoformat() if a.created_at else None,
                        "name": name})
    return {"analyses": out}
