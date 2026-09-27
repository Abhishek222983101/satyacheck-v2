"""P8.4 — Chain Theatre endpoint: ForwardChain gens 1..k of an uploaded clip, each analysed in turn.
Emits gen_ready + per-gen verdict events on one SSE stream."""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from sqlmodel import Session

from server import runner
from server.db import Analysis, Artifact, Chain, ChainGen, Media, engine, j
from server.util import OUT, check_id

router = APIRouter(prefix="/v1")

from ml.forwardchain.emulate import emulate


class ChainReq(BaseModel):
    media_id: str
    gens: int = 5


def _run_chain(chain_id: str, case_id: str, src: Path, gens: int, loop: asyncio.AbstractEventLoop):
    def emit(kind: str, data: dict):
        loop.call_soon_threadsafe(asyncio.create_task, runner.emit(chain_id, kind, data))

    try:
        recs = emulate(str(src), str(OUT / case_id / chain_id[:8]), gens=gens, seed=abs(hash(chain_id)) % (2**31))
        for rec in recs:
            media_id = analysis_id = str(uuid.uuid4())
            with Session(engine) as s:
                s.add(Media(id=media_id, case_id=case_id, kind="gen", gen=rec["gen"],
                            path=rec["path"], sha256="", duration_s=rec["probe"]["duration"],
                            width=rec["probe"]["width"], height=rec["probe"]["height"],
                            has_audio=rec["probe"]["audio"] is not None, emulator_params=j(rec["params"])))
                s.flush()
                s.add(Analysis(id=analysis_id, media_id=media_id, status="queued", model_versions=j(runner.MODELS)))
                s.add(ChainGen(id=str(uuid.uuid4()), chain_id=chain_id, gen=rec["gen"],
                               media_id=media_id, analysis_id=analysis_id))
                s.add(Artifact(id=str(uuid.uuid4()), analysis_id=analysis_id, name=f"gen_{rec['gen']}.mp4",
                               path=rec["path"], mime="video/mp4"))
                s.commit()
            emit("gen_ready", {"gen": rec["gen"], "media_url": f"/v1/analyses/{analysis_id}/artifacts/gen_{rec['gen']}.mp4",
                               "params": rec["params"]})
            result = runner.run_analysis_sync(analysis_id, rec["path"],
                                              on_event=lambda k, d, g=rec["gen"]: emit(k, {"gen": g, **d} if k == "stage" else d))
            emit("branch", {"branch": "verdict", "gen": rec["gen"], "result": result["verdict"]})
        emit("end", {})
    except Exception as e:
        emit("error", {"type": e.__class__.__name__, "title": "Chain failed",
                       "detail": "The forwarding simulation hit a snag — sorry! Please try again."})
    finally:
        with Session(engine) as s:
            c = s.get(Chain, chain_id)
            if c:
                c.status = "done"
                s.add(c); s.commit()


@router.get("/chains/{cid}")
async def chain_status(cid: str):
    """Reconciler for the Chain page: full gen state in one poll, so the UI converges
    even if the SSE stream drops (refresh mid-run also recovers)."""
    check_id(cid)
    with Session(engine) as s:
        c = s.get(Chain, cid)
        if not c:
            from server.util import problem
            return problem(404, "Not found", "No such chain.")
        from sqlmodel import select
        gens = s.exec(select(ChainGen).where(ChainGen.chain_id == cid).order_by(ChainGen.gen)).all()
        out = []
        for g in gens:
            a = s.get(Analysis, g.analysis_id)
            label, p = None, None
            if a and a.result_json:
                import json as _j
                v = _j.loads(a.result_json)["verdict"]
                label, p = v["label"], v["p_fake"]
            out.append({"gen": g.gen, "media_id": g.media_id, "analysis_id": g.analysis_id,
                        "media_url": f"/v1/analyses/{g.analysis_id}/artifacts/gen_{g.gen}.mp4",
                        "status": a.status if a else "queued", "label": label, "p_fake": p})
        return {"chain_id": cid, "status": c.status, "gens": out}


@router.post("/chains", status_code=202)
async def create_chain(req: ChainReq):
    check_id(req.media_id)
    if not 1 <= req.gens <= 5:
        raise HTTPException(422, "gens must be 1..5")
    with Session(engine) as s:
        m = s.get(Media, req.media_id)
        if not m:
            raise HTTPException(404, "no such media")
    chain_id = str(uuid.uuid4())
    with Session(engine) as s:
        s.add(Chain(id=chain_id, case_id=m.case_id, source_media_id=m.id))
        s.commit()
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _run_chain, chain_id, m.case_id, Path(m.path), req.gens, loop)
    return {"chain_id": chain_id, "events_url": f"/v1/analyses/{chain_id}/events"}
