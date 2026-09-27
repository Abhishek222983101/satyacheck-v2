"""Runs satya_core pipeline in worker threads, streams SSE events, adapts output to
the AnalysisResult v2 contract (contracts/analysis.schema.json).

Real pipeline loads torch models lazily on first use. If weights/GPU are unavailable,
SATYA_MOCK=1 (or load failure) swaps in a mock that produces contract-valid results so
the server, frontend and tests run everywhere. The swap point is exactly one function.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

MODELS: dict[str, str] = {"syncformer": "syncformer_best.pt", "spliceloc": "spliceloc_v2/best.pt",
                          "facetrace": "facetrace_v2/best.pt", "forwardage": "forwardage_lgbm",
                          "nyaya": "nyaya.pt"}

_events: dict[str, list[dict]] = {}   # analysis_id -> persisted event log
_queues: dict[str, set[asyncio.Queue]] = {}  # analysis_id -> live SSE subscribers
_lock = asyncio.Lock()


async def emit(analysis_id: str, event: str, data: dict) -> None:
    async with _lock:
        log = _events.setdefault(analysis_id, [])
        log.append({"event": event, "data": data})
        if len(log) > 2000:  # bounded: a stuck stream can't grow memory forever
            del log[:len(log) - 2000]
        for q in _queues.get(analysis_id, ()):
            q.put_nowait({"event": event, "data": data})


def subscribe(analysis_id: str) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue()
    _queues.setdefault(analysis_id, set()).add(q)
    return q


def unsubscribe(analysis_id: str, q: asyncio.Queue) -> None:
    _queues.get(analysis_id, set()).discard(q)


def replay(analysis_id: str) -> list[dict]:
    return list(_events.get(analysis_id, []))


# --------------------------------------------------------------------------- pipeline

class Pipeline:
    """Lazy singleton around satya_core. Falls back to mock; reports which mode is live."""

    def __init__(self) -> None:
        self.mock = os.environ.get("SATYA_MOCK") == "1"
        self.runtime = os.environ.get("SATYA_RUNTIME", "torch")  # torch | ort
        self._impl = None

    def _load(self):
        if self._impl is not None or self.mock:
            return self._impl
        try:
            if self.runtime == "ort":
                from satya_core.onnx_models import OnnxModels
                from satya_core.verdict import Fuser
                self._impl = {"models": OnnxModels(), "fuser": Fuser("models/nyaya.pt")}
            else:
                from satya_core.pipeline import Models
                from satya_core.verdict import Fuser
                dev = os.environ.get("SATYA_DEVICE", "cuda")
                self._impl = {"models": Models(device=dev), "fuser": Fuser("models/nyaya.pt")}
        except Exception as e:  # weights missing, no GPU, cold cache: degrade, never crash the API
            print(f"[runner] real pipeline unavailable ({e.__class__.__name__}: {e}); mock mode")
            self.mock = True
        return self._impl

    def warm(self) -> bool:
        return self._load() is not None


PIPE = Pipeline()


def _mock_branches(meta: dict) -> dict:
    no_face, no_speech = meta.get("no_face"), meta.get("no_speech") or not meta.get("has_audio")
    r_face = {"status": "abstained", "reason": "no_face"} if no_face else None
    r_speech = {"status": "abstained", "reason": "no_speech"} if no_speech else None
    return {
        "sync": r_face or r_speech or {"status": "ok", "clip_score": 0.18, "offset_ms": 0.0, "spans": []},
        "splice": r_speech or {"status": "ok", "p_spliced": 0.11, "p_synthetic": 0.07, "frame_spoof": [0.05] * 8, "boundaries": []},
        "face": r_face or {"status": "ok", "clip_score": 0.22, "frame_probs": [0.2, 0.24]},
        "age": {"status": "ok", "gen_hat": 0.4, "gen_dist": [0.7, 0.25, 0.05, 0, 0, 0], "platform": {}, "why": ["mock mode"]},
    }


_HEADLINES = {
    "REAL": ("Looks genuine to me — I checked twice, promise. \U0001f49b",
             "यह असली लगती है — मैंने दो बार जाँचा, वादा।"),
    "FAKE": ("I'm so sorry… this clip has been doctored.",
             "माफ़ कीजिए… यह क्लिप बदली गई है।"),
    "SUSPECT": ("I'm not sure about this one — sorry! Treat it with care.",
                "मुझे पूरा यकीन नहीं — क्षमा करें! सावधानी से देखें।"),
}


def _headlines(label: str, gen_hat: float) -> tuple[str, str]:
    en, hi = _HEADLINES[label]
    if label == "REAL" and gen_hat >= 3.5:
        en = "Looks genuine, though it's been forwarded a lot — still fine."
        hi = "यह असली लगती है, हालाँकि काफ़ी फॉरवर्ड हुई है — फिर भी ठीक है।"
    return en, hi


def _downsample(arr: list | None, cap: int = 600) -> list:
    arr = [round(float(x), 4) for x in (arr or [])]
    if len(arr) <= cap:
        return arr
    step = len(arr) / cap
    return [arr[int(i * step)] for i in range(cap)]


def _adapt(run: dict, media_row, mode: str) -> dict:
    """satya_core output -> AnalysisResult v2 dict, exactly the contract's shape."""
    b, meta, v = run["branches"], run["meta"], run.get("verdict") or {}
    gen_hat = float(b.m3.get("gen_hat") or 0.0)
    m1, m2, m4, m3 = b.m1, b.m2, b.m4, b.m3
    verdict = {
        "label": v.get("label", "SUSPECT"), "p_fake": round(float(v.get("p_fake", 0.5)), 4),
        "integrity": int(v.get("integrity", 50)), "conformal_set": v.get("conformal_set", ["REAL", "FAKE"]),
        "alpha": float(v.get("alpha", 0.05)), "bucket": v.get("bucket", "gen_0_1"),
        "reason": v.get("reason"), **dict(zip(("headline_en", "headline_hi"), _headlines(v.get("label", "SUSPECT"), gen_hat))),
    }
    branches = {
        "sync": {"status": m1["status"], "reason": m1.get("reason"),
                 **({"p_fake": round(float(m1["clip_score"]), 4), "offset_ms": float(m1["offset_ms"]),
                     "curve": _downsample(m1.get("sync_prob") and m1["sync_prob"].mean(0).tolist()),
                     "spans": [{"t0": round(t0, 2), "t1": round(t1, 2), "score": round(s, 4)} for t0, t1, s in m1.get("spans", [])]}
                    if m1["status"] == "ok" else {})},
        "splice": {"status": m2["status"], "reason": m2.get("reason"),
                   **({"p_spliced": round(float(m2["p_spliced"]), 4), "p_synthetic": round(float(m2["p_synthetic"]), 4),
                       "frame_spoof": _downsample(m2.get("frame_spoof")),
                       "boundaries": [{"t": round(x["t"], 3), "strength": round(x["strength"], 4)} for x in m2.get("boundaries", [])]}
                      if m2["status"] == "ok" else {})},
        "face": {"status": m4["status"], "reason": m4.get("reason"),
                 **({"p_fake": round(float(m4["clip_score"]), 4), "frames": _downsample(m4.get("frame_probs")), "cam_refs": []}
                    if m4["status"] == "ok" else {})},
        "age": {"status": m3["status"], "gen_hat": round(gen_hat, 3),
                "gen_dist": [round(float(x), 4) for x in (m3.get("gen_dist") or [0] * 6)],
                "platform": {}, "why": m3.get("why", [])},
    }
    for br in branches.values():  # contract: omit reason when null (enum forbids null)
        if br.get("reason") is None:
            br.pop("reason", None)
    out = {
        "schema_version": "2.0", "id": media_row.id,
        "file": {"name": Path(media_row.path).name, "sha256": media_row.sha256,
                 "duration_s": round(float(meta.get("duration_s") or media_row.duration_s or 0), 2),
                 "assessed_s": round(float(meta.get("assessed_s") or 0), 2),
                 "width": meta.get("width", media_row.width), "height": meta.get("height", media_row.height),
                 "has_audio": bool(meta.get("has_audio"))},
        "verdict": verdict,
        "branches": branches,
        "timeline": {"fps": meta.get("fps"), "speech": _downsample(meta.get("speech")), "mouth_open": _downsample(meta.get("mouth_open"))},
        "timings_ms": {k: float(x) for k, x in run["timings_ms"].items()},
        "models": {k: hashlib.sha256(v.encode()).hexdigest()[:12] for k, v in MODELS.items()},
        "mode": mode,
    }
    return out


def run_analysis_sync(analysis_id: str, media_path: str, on_event=None) -> dict:
    """Blocking; call from a worker thread. Emits stage/branch/verdict SSE events; returns the contract dict."""
    t0 = time.time()

    def stage(name: str, progress: float | None = None):
        ev = {"stage": name, "t_ms": round((time.time() - t0) * 1000), "progress": progress}
        if on_event:
            on_event("stage", ev)

    impl = PIPE._load()
    if impl is None:  # mock
        from types import SimpleNamespace
        for name, prog in (("decode", 0.18), ("landmarks", 0.36), ("age", 0.54), ("face", 0.68), ("sync", 0.82), ("splice", 0.92)):
            time.sleep(0.25)
            if on_event:
                on_event("stage", {"stage": name, "t_ms": int((time.time() - t0) * 1000), "progress": prog})
        time.sleep(0.3)  # let the UI rail move; mock stays honest about being a mock
        meta = {"duration_s": 8.0, "assessed_s": 8.0, "has_audio": True, "width": 640, "height": 360,
                "face_presence": 0.9, "speech_ratio": 0.7, "speech": [0.5] * 50, "mouth_open": [0.3] * 50, "fps": 25.0}
        mb = _mock_branches(meta)
        branches = SimpleNamespace(m1=mb["sync"], m2=mb["splice"], m4=mb["face"], m3=mb["age"], timings_ms={}, meta={})
        p = 0.17
        verdict = {"label": "REAL", "p_fake": p, "integrity": round(100 * (1 - p)), "conformal_set": ["REAL"],
                   "alpha": 0.05, "bucket": "gen_0_1", "reason": None}
        run = {"branches": branches, "meta": meta, "timings_ms": {"total": 300, "mock": 300}, "verdict": verdict}
    else:
        from satya_core.pipeline import analyze_path
        run = analyze_path(impl["models"], media_path, fuser=impl["fuser"], on_stage=stage)

    from server.db import Analysis, Media, Session, engine
    with Session(engine) as s:
        m = s.get(Media, analysis_id)  # analysis id == media id for originals
        a = s.get(Analysis, analysis_id)
        if m is None or a is None:
            raise RuntimeError(f"rows missing for {analysis_id}")
        result = _adapt(run, m, mode="live" if impl else "mock")
        a.status, a.label, a.p_fake = "done", result["verdict"]["label"], result["verdict"]["p_fake"]
        a.reason, a.bucket = result["verdict"]["reason"], result["verdict"]["bucket"]
        a.result_json, a.total_ms = json.dumps(result), int(result["timings_ms"]["total"])
        a.model_versions, a.finished_at = json.dumps(result["models"]), datetime.now(timezone.utc)
        s.add(a); s.commit()
    if on_event:
        on_event("verdict", {**result["verdict"], "timings_ms": result["timings_ms"]})
        on_event("branch", {"branch": "age", "result": result["branches"]["age"]})
    return result
