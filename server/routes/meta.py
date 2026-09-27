"""Meta endpoints: health, benchmarks (serves ml/reports JSONs), atlas stub."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from server import runner
from server.util import problem

router = APIRouter(prefix="/v1")

REPORTS = (Path(__file__).resolve().parents[2] / "ml" / "reports")
ALLOWED_REPORTS = {"heldout_wav2lip", "nyaya", "spliceloc", "forwardage", "ablations", "benchmark"}


@router.get("/health")
async def health():
    warm = runner.PIPE.warm()
    return {"ok": True, "mode": "mock" if runner.PIPE.mock else "live", "warm": warm,
            "models": [{"name": k, "sha": None} for k in runner.MODELS], "cpu_threads": 4}


@router.get("/benchmarks/{name}")
async def benchmarks(name: str):
    if name not in ALLOWED_REPORTS:
        return problem(404, "Not found", "No such benchmark report.")
    p = REPORTS / f"{name}.json"
    if not p.exists():
        return problem(404, "Not found", "That report hasn't been generated yet.")
    return JSONResponse(json.loads(p.read_text()))


@router.get("/benchmarks")
async def all_benchmarks():
    out = {}
    for name in sorted(ALLOWED_REPORTS):
        p = REPORTS / f"{name}.json"
        if p.exists():
            out[name] = json.loads(p.read_text())
    return out
