"""Intake for user recordings (M1 webcam top-up + M2 opus fine-tune data).

Collects candidate clips from:
  1. data/user_recordings/inbox/            (user drops files here)
  2. var/uploads/*/  whose Case.source=='webcam'   (in-app "Record yourself" uploads)

For each: dedupe by content hash, run a cheap quality gate (face present, speech present,
>=5 s, <=60 s), transcode to a uniform mp4, and append to
data/manifests/user_recordings.parquet  (path, sha, seconds, speech_ratio, face_presence, lang).

Every clip in that manifest is a GENUINE recording by definition (recorded by the user,
uploaded straight from the capture UI or dropped from their own camera roll) — that is the
ground-truth label the adaptation trains on. No synthetic fakes are mixed in.

Usage:  python -m ml.data.harvest_recordings            # harvest + quality-gate + manifest
        python -m ml.data.harvest_recordings --stats    # just show what's collected
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

INBOX = Path("data/user_recordings/inbox")
STORE = Path("data/user_recordings/store")       # accepted, transcoded clips
REJECT = Path("data/user_recordings/rejected")   # gate failures (kept for inspection)
MANIFEST = Path("data/manifests/user_recordings.parquet")


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def probe(p: Path) -> dict | None:
    r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams",
                        "-show_format", str(p)], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        j = json.loads(r.stdout)
        v = next((s for s in j["streams"] if s.get("codec_type") == "video"), None)
        a = next((s for s in j["streams"] if s.get("codec_type") == "audio"), None)
        dur = float(j.get("format", {}).get("duration") or (v or {}).get("duration") or 0)
        return {"dur": dur, "has_video": v is not None, "has_audio": a is not None}
    except Exception:
        return None


def quality_gate(p: Path) -> tuple[bool, str, dict]:
    info = probe(p)
    if not info:
        return False, "unreadable", {}
    if not info["has_video"] or not info["has_audio"]:
        return False, "needs both video and audio", info
    if not 5 <= info["dur"] <= 60:
        return False, f"duration {info['dur']:.1f}s (want 5-60)", info
    from satya_core.preprocess import preprocess
    try:
        pre = preprocess(str(p), vad=True)
    except Exception as e:
        return False, f"decode failed: {e.__class__.__name__}", info
    meta = pre.meta
    if meta.get("no_face"):
        return False, "no face detected", info
    if meta.get("no_speech"):
        return False, "no speech detected", info
    return True, "ok", {**info, "speech_ratio": meta.get("speech_ratio"),
                        "face_presence": meta.get("face_presence")}


def collect_candidates() -> list[Path]:
    cands = [p for p in INBOX.iterdir() if p.suffix.lower() in (".webm", ".mp4", ".mov", ".mkv", ".opus")] if INBOX.exists() else []
    # in-app webcam uploads: join media->case to filter source
    try:
        from sqlmodel import Session, select
        from server.db import Case, Media, engine
        with Session(engine) as s:
            rows = s.exec(select(Media, Case).where(Media.case_id == Case.id, Case.source == "webcam",
                                                    Media.kind == "original")).all()  # gens are re-encodes, not ground truth
            cands += [Path(m.path) for m, _c in rows if Path(m.path).exists()]
    except Exception as e:
        print(f"(server db unavailable: {e.__class__.__name__} — inbox only)")
    seen, out = set(), []
    for p in cands:
        h = sha16(p)
        if h not in seen:
            seen.add(h)
            out.append(p)
    return out


def main() -> None:
    stats_only = "--stats" in sys.argv
    STORE.mkdir(parents=True, exist_ok=True)
    REJECT.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    old = pd.read_parquet(MANIFEST) if MANIFEST.exists() else pd.DataFrame(columns=["sha", "path", "dur_s", "speech_ratio", "face_presence"])
    known = set(old.sha)

    cands = collect_candidates()
    print(f"{len(cands)} unique candidate clips")
    if stats_only:
        print(f"manifest already has {len(old)} accepted clips")
        return

    rows = [dict(r) for r in old.to_dict("records")]
    for p in cands:
        h = sha16(p)
        if h in known:
            continue
        ok, why, info = quality_gate(p)
        if not ok:
            print(f"  reject {p.name[:40]:40s} {why}")
            shutil.move(str(p), REJECT / p.name)
            continue
        # transcode to uniform clean mp4 (audio: the mic-ingest normalisation chain)
        dst = STORE / f"rec_{h}.mp4"
        r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(p),
                            "-map", "0:v:0", "-map", "0:a:0", "-vf", "format=yuv420p",
                            "-af", "afftdn=nr=10:nf=-20", "-c:v", "libx264", "-preset", "veryfast",
                            "-crf", "23", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart",
                            str(dst)], capture_output=True)
        if r.returncode != 0 or not dst.exists():
            print(f"  reject {p.name[:40]:40s} transcode failed")
            continue
        rows.append({"sha": h, "path": str(dst), "dur_s": round(info["dur"], 2),
                     "speech_ratio": round(float(info.get("speech_ratio") or 0), 3),
                     "face_presence": round(float(info.get("face_presence") or 0), 3)})
        known.add(h)
        print(f"  accept {p.name[:40]:40s} {info['dur']:.1f}s speech {rows[-1]['speech_ratio']}")

    pd.DataFrame(rows).to_parquet(MANIFEST, index=False)
    print(f"manifest: {len(rows)} clips -> {MANIFEST}")
    print("next: when you have 10+ accepted clips, say 'run the adaptation' — M2 opus fine-tune + M1 webcam top-up + Nyaya refit")


if __name__ == "__main__":
    main()
