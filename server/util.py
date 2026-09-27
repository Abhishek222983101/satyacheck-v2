"""Shared request-handling helpers: strict upload validation and problem+json errors."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import magic
from fastapi import HTTPException, UploadFile
from fastapi.responses import JSONResponse

from server.db import VAR

UPLOADS, OUT = VAR / "uploads", VAR / "out"
MAX_UPLOAD_MB = 100
BLOCKED_MIMES = {"application/x-executable", "application/x-dosexec", "application/zip",
                 "application/x-sh", "text/html", "application/javascript"}
NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,80}$")
_id = re.compile(r"^[0-9a-fA-F-]{1,64}$")


def problem(status: int, title: str, detail: str, type_: str = "about:blank") -> JSONResponse:
    return JSONResponse({"type": type_, "title": title, "detail": detail, "status": status},
                        status_code=status, media_type="application/problem+json")


def check_id(x: str) -> str:
    if not _id.fullmatch(x):
        raise HTTPException(400, "bad id")
    return x


def safe_artifact(name: str) -> str:
    """Artifact names come from the URL: whitelist the charset, never trust a path."""
    if not NAME_RE.fullmatch(name):
        raise HTTPException(400, "bad artifact name")
    return name


async def save_upload(f: UploadFile, case_id: str) -> dict:
    """Stream to disk with a hard size cap; sniff content via libmagic (never trust
    the declared type); then ffprobe as the authoritative gate.
    Returns {path, sha256, duration_s, width, height, has_audio}."""
    OUT_CASE = UPLOADS / case_id
    OUT_CASE.mkdir(parents=True, exist_ok=True)
    stem = (f.filename or "clip")[-60:]
    dst = OUT_CASE / (hashlib.sha256(stem.encode()).hexdigest()[:16] + Path(stem).suffix[-6:])
    size, sha = 0, hashlib.sha256()
    with dst.open("wb") as w:
        while chunk := await f.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_MB << 20:
                dst.unlink(missing_ok=True)
                raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB — please trim it first.")
            sha.update(chunk)
            w.write(chunk)
    head = dst.open("rb").read(1 << 16)
    mime = magic.from_buffer(head, mime=True)
    if mime in BLOCKED_MIMES:
        dst.unlink(missing_ok=True)
        raise HTTPException(415, f"That file type ({mime}) isn't a clip we can read. Upload an MP4, WebM, MOV, MP3 or WAV.")
    probe = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(dst)],
                           capture_output=True, text=True)
    if probe.returncode != 0:
        dst.unlink(missing_ok=True)
        raise HTTPException(415, "This file has no readable stream. It may be corrupted — try re-exporting it.")
    info = json.loads(probe.stdout or "{}")
    streams = info.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = info.get("format", {})
    dur = float(fmt.get("duration") or (v or {}).get("duration") or 0)
    return {"path": str(dst), "sha256": sha.hexdigest(), "duration_s": dur,
            "width": int(v["width"]) if v else None, "height": int(v["height"]) if v else None,
            "has_audio": a is not None}
