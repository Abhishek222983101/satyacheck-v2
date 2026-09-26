"""ForwardChain — WhatsApp generational-loss emulator (blueprint §4.2.2).

Each *hop* is one WhatsApp forward: downscale -> H.264 re-encode at high CRF -> AAC re-encode -> re-mux,
with per-hop parameters drawn from a distribution. Gen-k = k hops applied in sequence.

Parameters come from PRIORS until `whatsapp_2026.json` (fitted from real forwards, P1.3) exists; then
that file wins. Deterministic for a given (seed, hop index).

CLI:  python -m ml.forwardchain.emulate in.mp4 --gens 5 --out dir/ --seed 7
"""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

FITTED = Path(__file__).with_name("whatsapp_2026.json")

# Literature/ffprobe priors (UC Denver WhatsApp forensics theses; SNVSE method). Replaced by FITTED when present.
PRIORS = {
    "long_side": {"values": [640, 848, 480], "p": [0.5, 0.3, 0.2]},
    "crf": {"lo": 28.0, "hi": 34.0},
    "preset": {"values": ["veryfast", "faster", "fast"], "p": [0.4, 0.4, 0.2]},
    "profile": {"values": ["baseline", "main"], "p": [0.35, 0.65]},
    "gop": {"lo": 30, "hi": 250},
    "fps": {"values": [25, 30, 24], "p": [0.35, 0.5, 0.15]},
    "p_vfr": 0.3,
    "aac_kbps": {"values": [48, 64, 96], "p": [0.3, 0.4, 0.3]},
    "audio_sr": {"values": [44100, 48000], "p": [0.5, 0.5]},
    "p_mono": 0.4,
    "p_rerecord": 0.08,
    "p_overlay": 0.05,
}


@dataclass
class HopParams:
    long_side: int
    crf: float
    preset: str
    profile: str
    gop: int
    fps: int
    vfr: bool
    aac_kbps: int
    audio_sr: int
    mono: bool
    rerecord: bool
    overlay: bool
    scaler: str = "bicubic"
    extra: dict = field(default_factory=dict)


def load_priors() -> dict:
    if FITTED.exists():
        return json.loads(FITTED.read_text())
    return PRIORS


def draw_hop(rng: np.random.Generator, priors: dict | None = None) -> HopParams:
    P = priors or load_priors()
    pick = lambda k: P[k]["values"][rng.choice(len(P[k]["values"]), p=P[k]["p"])]
    return HopParams(
        long_side=int(pick("long_side")),
        crf=float(rng.uniform(P["crf"]["lo"], P["crf"]["hi"])),
        preset=str(pick("preset")),
        profile=str(pick("profile")),
        gop=int(rng.integers(P["gop"]["lo"], P["gop"]["hi"] + 1)),
        fps=int(pick("fps")),
        vfr=bool(rng.random() < P["p_vfr"]),
        aac_kbps=int(pick("aac_kbps")),
        audio_sr=int(pick("audio_sr")),
        mono=bool(rng.random() < P["p_mono"]),
        rerecord=bool(rng.random() < P["p_rerecord"]),
        overlay=bool(rng.random() < P["p_overlay"]),
        scaler=str(rng.choice(["bicubic", "lanczos", "bilinear"])),
    )


def probe(path: str | Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    j = json.loads(out)
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    a = next((s for s in j["streams"] if s["codec_type"] == "audio"), None)
    dur = float(j["format"].get("duration") or v.get("duration") or 0)
    size = int(j["format"]["size"])
    return {
        "width": int(v["width"]), "height": int(v["height"]), "codec": v["codec_name"],
        "profile": v.get("profile"), "fps": v.get("avg_frame_rate"), "duration": dur,
        "kbps_total": round(size * 8 / max(dur, 1e-6) / 1000, 1),
        "audio": None if a is None else {
            "codec": a["codec_name"], "sr": int(a["sample_rate"]), "ch": int(a["channels"]),
            "kbps": round(int(a.get("bit_rate") or 0) / 1000, 1),
        },
    }


def _even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


def hop(src: str | Path, dst: str | Path, p: HopParams, has_audio: bool | None = None) -> None:
    """Apply one WhatsApp-style forward to src -> dst."""
    info = probe(src)
    if has_audio is None:
        has_audio = info["audio"] is not None
    w, h = info["width"], info["height"]
    long_in = max(w, h)
    long_out = min(p.long_side, long_in)  # WhatsApp never upscales
    s = long_out / long_in
    ow, oh = _even(w * s), _even(h * s)

    vf = []
    if p.rerecord:  # re-captured on another phone: small crop, gamma shift, sensor-ish noise
        vf += ["crop=iw*0.95:ih*0.95", "eq=gamma=1.06:contrast=0.97", "noise=alls=3:allf=t"]
    vf.append(f"scale={ow}:{oh}:flags={p.scaler}")
    if p.overlay:  # burned-in caption bar
        vf.append("drawbox=x=0:y=ih*0.86:w=iw:h=ih*0.09:color=black@0.55:t=fill")
    vf.append("format=yuv420p")

    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-map", "0:v:0"]
    if has_audio:
        cmd += ["-map", "0:a:0"]
    cmd += ["-vf", ",".join(vf)]
    # ffmpeg >= 7 rejects -r together with a non-CFR fps_mode: VFR keeps source timestamps, CFR forces the rate.
    cmd += ["-fps_mode", "vfr"] if p.vfr else ["-r", str(p.fps), "-fps_mode", "cfr"]
    cmd += ["-c:v", "libx264", "-preset", p.preset, "-profile:v", p.profile, "-crf", f"{p.crf:.1f}",
            "-g", str(p.gop), "-bf", "0" if p.profile == "baseline" else "2", "-threads", "2"]
    if has_audio:
        cmd += ["-c:a", "aac", "-b:a", f"{p.aac_kbps}k", "-ar", str(p.audio_sr), "-ac", "1" if p.mono else "2"]
    else:
        cmd += ["-an"]
    cmd += ["-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True)


def emulate(src: str | Path, out_dir: str | Path, gens: int = 5, seed: int = 0,
            priors: dict | None = None) -> list[dict]:
    """Produce gen_1..gen_k by chaining hops. Returns per-gen records (also written to params.json)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    P = priors or load_priors()
    has_audio = probe(src)["audio"] is not None
    recs, cur = [], Path(src)
    for k in range(1, gens + 1):
        rng = np.random.default_rng([seed, k])
        p = draw_hop(rng, P)
        dst = out / f"gen_{k}.mp4"
        hop(cur, dst, p, has_audio=has_audio)
        recs.append({"gen": k, "path": str(dst), "params": asdict(p), "probe": probe(dst)})
        cur = dst
    (out / "params.json").write_text(json.dumps(recs, indent=2))
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src")
    ap.add_argument("--gens", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for r in emulate(a.src, a.out, a.gens, a.seed):
        pr = r["probe"]
        au = pr["audio"]
        print(f"gen {r['gen']}: {pr['width']}x{pr['height']}  {pr['kbps_total']:>7} kbps  crf {r['params']['crf']:.1f}"
              f"  audio {'—' if au is None else str(au['kbps']) + 'kbps/' + str(au['sr']) + 'Hz/' + str(au['ch']) + 'ch'}")


if __name__ == "__main__":
    main()
