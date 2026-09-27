"""P8.4 — Red team endpoints: Whisper word list + audio splice attacks (delete/swap/duplicate).
Splice runs the measured P1.5 factory (ml.redteam.splice) on extracted audio; the result is
muxed back over the original video so the attack is a real playable clip."""
from __future__ import annotations

import asyncio
import subprocess
import uuid
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlmodel import Session

from server.db import Analysis, Artifact, Attack, Media, engine, j
from server.util import OUT, check_id

router = APIRouter(prefix="/v1")


def _load_wav(path: str) -> tuple[np.ndarray, int]:
    import av
    rs = av.AudioResampler(format="fltp", layout="mono", rate=16000)
    chunks = []
    with av.open(path) as c:
        for pkt in c.demux(audio=0):
            for fr in pkt.decode():
                for g in rs.resample(fr):
                    chunks.append(g.to_ndarray()[0])
    if not chunks:
        raise HTTPException(422, "This clip has no audio track, so there are no words to edit.")
    return np.concatenate(chunks).astype(np.float32), 16000


@router.get("/redteam/words/{media_id}")
async def words(media_id: str):
    check_id(media_id)
    with Session(engine) as s:
        m = s.get(Media, media_id)
        if not m:
            raise HTTPException(404, "no such media")
    from ml.redteam.words import pick_units
    wav, _ = _load_wav(m.path)
    out = await asyncio.get_running_loop().run_in_executor(None, pick_units, wav, None)
    return out  # {"words": [{w,t0,t1,conf}], "fallback": bool}


class SpliceReq(BaseModel):
    media_id: str
    op: str  # delete | swap | duplicate
    t0: float
    t1: float


@router.post("/redteam/splice")
async def splice(req: SpliceReq):
    check_id(req.media_id)
    if req.op not in ("delete", "swap", "duplicate"):
        raise HTTPException(422, "op must be delete, swap or duplicate")
    if not (0 <= req.t0 < req.t1 <= 600) or req.t1 - req.t0 > 30:
        raise HTTPException(422, "span must satisfy 0 <= t0 < t1 <= 600 and be shorter than 30 s")
    with Session(engine) as s:
        m = s.get(Media, req.media_id)
        if not m:
            raise HTTPException(404, "no such media")
        src, case_id = Path(m.path), m.case_id

    from ml.redteam.splice import splice_spans
    wav, sr = _load_wav(src)
    loop = asyncio.get_running_loop()
    res = await loop.run_in_executor(None, lambda: splice_spans(wav, sr, req.op, (req.t0, req.t1)))

    out_dir = OUT / case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    attack_id = str(uuid.uuid4())
    wav_path = out_dir / f"attack_{attack_id[:8]}.wav"
    clip_path = out_dir / f"attack_{attack_id[:8]}.mp4"
    import soundfile as sf
    sf.write(wav_path, res.audio, sr)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-i", str(wav_path),
                    "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest", str(clip_path)], check=True)
    wav_path.unlink(missing_ok=True)

    media_id = str(uuid.uuid4())
    with Session(engine) as s:
        s.add(Media(id=media_id, case_id=case_id, kind="attack", path=str(clip_path), sha256="",
                    duration_s=res.duration, has_audio=True))
        s.flush()
        s.add(Attack(id=attack_id, case_id=case_id, kind=f"splice_{req.op}",
                     params=j({"op": req.op, "t0": req.t0, "t1": req.t1, "boundaries": res.boundaries}),
                     input_media_id=req.media_id, output_media_id=media_id, status="done"))
        s.flush()
        s.add(Analysis(id=str(uuid.uuid4()), media_id=media_id, status="queued", model_versions=j({})))
        s.add(Artifact(id=str(uuid.uuid4()), analysis_id=media_id, name="attack.mp4", path=str(clip_path), mime="video/mp4"))
        s.commit()
    return {"attack_id": attack_id, "media_id": media_id,
            "media_url": f"/v1/analyses/{media_id}/artifacts/attack.mp4",
            "truth": {"boundaries": [round(b, 3) for b in res.boundaries]}}
