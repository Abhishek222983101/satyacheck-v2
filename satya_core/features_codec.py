"""Codec-domain forensics features for ForwardAge (M3) — blueprint §4.6. Cheap (<0.6 s / 10 s clip), no GPU.

Every WhatsApp forward leaves a fingerprint that a re-encode cannot fully hide: lower bits/pixel, coarser quantisation
(sparser AC coefficients, DCT first-digit statistics drift from Benford), a stronger 8x8 block grid, a lower audio
bandwidth cutoff (each AAC pass shaves the top of the spectrum), a different GOP/B-frame structure and container.
`extract(path)` returns (vector float32, names). Order is fixed: models are trained against `NAMES`.
"""
from __future__ import annotations

import json
import subprocess

import av
import cv2
import numpy as np
from scipy.signal import welch

_PT = {1: "I", 2: "P", 3: "B"}   # AVPictureType (PyAV >= 18 exposes it as int)
BENFORD = np.log10(1 + 1 / np.arange(1, 10))


def _probe(path):
    j = json.loads(subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path],
                                  capture_output=True, text=True, check=True).stdout)
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    a = next((s for s in j["streams"] if s["codec_type"] == "audio"), None)
    return j, v, a


def _benford(coefs: np.ndarray):
    """First-digit distribution (9-d) of non-zero |coefficients| + KL-ish divergence from Benford + sparsity."""
    c = np.abs(coefs[coefs != 0])
    if len(c) < 50:
        return np.zeros(9), 0.0, 1.0
    d = (c / 10 ** np.floor(np.log10(c))).astype(int).clip(1, 9)
    h = np.bincount(d, minlength=10)[1:] / len(d)
    return h, float(np.sum(np.abs(h - BENFORD))), float(1 - len(c) / coefs.size)


def _dct_stats(gray_frames):
    """Blockwise DCT on I-frame luma; Benford digits per band + sparsity + block-grid ratio + Laplacian variance."""
    bands = {"lo": [(0, 1), (1, 0), (1, 1)], "mid": [(0, 2), (2, 0), (1, 2), (2, 1), (2, 2), (0, 3), (3, 0)],
             "hi": [(3, 3), (4, 4), (3, 4), (4, 3), (5, 5), (2, 5), (5, 2)]}
    acc = {k: [] for k in bands}
    grid_on = grid_off = 0.0; lap = []
    for g in gray_frames:
        g = cv2.resize(g, (320, 192), interpolation=cv2.INTER_AREA) if g.shape[1] > 320 else g
        h, w = (g.shape[0] // 8) * 8, (g.shape[1] // 8) * 8
        f = g[:h, :w].astype(np.float32) - 128
        blocks = f.reshape(h // 8, 8, w // 8, 8).swapaxes(1, 2).reshape(-1, 8, 8)
        D = np.stack([cv2.dct(b) for b in blocks[:: max(1, len(blocks) // 600)]])   # <=600 blocks per frame
        q = np.round(D / 2.0)                                                       # fixed coarse re-quantisation
        for k, pos in bands.items():
            acc[k].append(np.concatenate([q[:, i, j] for i, j in pos]))
        dx = np.abs(np.diff(f, axis=1))
        on = dx[:, 7::8].mean() if dx.shape[1] > 8 else 0.0
        off = np.delete(dx, np.arange(7, dx.shape[1], 8), axis=1).mean() if dx.shape[1] > 8 else 1.0
        grid_on += on; grid_off += off
        lap.append(cv2.Laplacian(g, cv2.CV_32F).var())
    out, div = [], []
    for k in bands:
        h, dv, sp = _benford(np.concatenate(acc[k])) if acc[k] else (np.zeros(9), 0.0, 1.0)
        out += list(h) + [sp]; div.append(dv)
    n = max(len(gray_frames), 1)
    return out + div + [grid_on / max(grid_off, 1e-6), float(np.log1p(np.mean(lap) if lap else 0))]


def _audio_bandwidth(path, max_s=6.0):
    """Native-rate spectrum -> cutoff frequency (-60 dB rel. peak, smoothed) — every AAC pass lowers it."""
    try:
        with av.open(path) as c:
            if not c.streams.audio:
                return [0.0, 0.0, 0.0, 0.0]
            s = c.streams.audio[0]; sr = s.rate; xs = []
            for fr in c.decode(s):
                xs.append(fr.to_ndarray().astype(np.float32).mean(0))
                if sum(map(len, xs)) > sr * max_s:
                    break
        x = np.concatenate(xs)
        if len(x) < 4096 or np.abs(x).max() < 1e-4:
            return [float(sr), 0.0, 0.0, 0.0]
        f, p = welch(x, sr, nperseg=2048)
        pdb = 10 * np.log10(p + 1e-14); pdb = np.convolve(pdb, np.ones(5) / 5, "same")
        ref = np.percentile(pdb[(f > 100) & (f < 3000)], 90)
        above = np.nonzero(pdb > ref - 60)[0]
        cutoff = float(f[above[-1]]) if len(above) else 0.0
        hf = float(np.mean(pdb[f > 0.75 * sr / 2]) - ref) if sr > 8000 else 0.0     # energy in the top quarter of the band
        return [float(sr), cutoff, cutoff / (sr / 2), hf]
    except Exception:
        return [0.0, 0.0, 0.0, 0.0]


NAMES: list[str] = []


def extract(path: str, max_s: float = 10.0):
    j, v, a = _probe(path)
    w, h = int(v["width"]), int(v["height"]); dur = float(j["format"].get("duration") or 1.0); dur = max(min(dur, 1e6), 0.1)
    fmt_br = float(j["format"].get("bit_rate") or 0)
    vbr = float(v.get("bit_rate") or 0) or fmt_br
    abr = float(a.get("bit_rate") or 0) if a else 0.0
    tags = j["format"].get("tags", {}) or {}
    brand = (tags.get("major_brand", "") or "").strip()
    enc = (tags.get("encoder", "") or "").lower() + (v.get("tags", {}) or {}).get("encoder", "").lower()
    r_fps = v.get("r_frame_rate", "0/1"); a_fps = v.get("avg_frame_rate", "0/1")
    fr = lambda s: (lambda n, d: n / d if d else 0)(*map(float, s.split("/")))
    profile = {"Baseline": 0, "Constrained Baseline": 0, "Main": 1, "High": 2}.get(v.get("profile", ""), 3)

    types, sizes, mv_mag, gray_i, ts = [], [], [], [], []
    with av.open(path) as c:
        vs = c.streams.video[0]; vs.thread_type = "AUTO"
        vs.codec_context.options = {"flags2": "+export_mvs"}
        for pkt in c.demux(vs):
            if pkt.dts is None:
                continue
            for f in pkt.decode():
                t = float(f.pts * vs.time_base) if f.pts is not None else len(ts) / 25
                if t > max_s:
                    break
                ts.append(t); types.append(_PT.get(int(f.pict_type), "?") if f.pict_type is not None else "?"); sizes.append(pkt.size)
                sd = f.side_data.get("MOTION_VECTORS") if hasattr(f.side_data, "get") else None
                if sd is not None:
                    m = sd.to_ndarray(); mv_mag.append(np.hypot(m["motion_x"], m["motion_y"]) / max(m["motion_scale"].max(), 1))
                if types[-1] == "I" and len(gray_i) < 4:
                    gray_i.append(f.to_ndarray(format="gray"))
            else:
                continue
            break
    ts = np.array(ts); n = max(len(ts), 1)
    tcount = {k: types.count(k) / n for k in ("I", "P", "B")}
    sizes = np.array(sizes, float) if sizes else np.zeros(1)
    kf_idx = [i for i, t in enumerate(types) if t == "I"]
    gop = float(np.mean(np.diff(kf_idx))) if len(kf_idx) > 1 else float(n)
    dts = np.diff(ts) if len(ts) > 2 else np.array([0.04])
    vfr = float(np.std(dts) / (np.mean(dts) + 1e-9))
    px = w * h
    bpp = vbr / max(fr(a_fps) or 25, 1) / px                    # bits per pixel per frame
    i_bits = float(np.mean([s for s, t in zip(sizes, types) if t == "I"] or [0])) * 8 / px
    p_bits = float(np.mean([s for s, t in zip(sizes, types) if t == "P"] or [0])) * 8 / px
    mv = np.concatenate([m.ravel() for m in mv_mag]) if mv_mag else np.zeros(1)
    mvh = np.histogram(np.clip(mv, 0, 16), bins=8, range=(0, 16))[0] / max(len(mv), 1)
    dct = _dct_stats(gray_i) if gray_i else [0.0] * 34
    aud = _audio_bandwidth(path)

    vec = [w, h, max(w, h), min(w, h), np.log(px), vbr / 1000, bpp, i_bits, p_bits, (i_bits + 1e-6) / (p_bits + 1e-6),
           profile, float(v.get("has_b_frames", 0)), tcount["I"], tcount["P"], tcount["B"], gop, vfr,
           fr(a_fps), fr(r_fps), float(brand.startswith("mp42")), float(brand.startswith("isom")), float("lavf" in enc or "lavc" in enc),
           float(len(mv_mag) > 0), float(np.mean(mv)), float(np.std(mv)), *mvh,
           *dct, abr / 1000, float(a["channels"]) if a else 0.0, float(a["sample_rate"]) if a else 0.0, *aud]
    global NAMES
    if not NAMES:
        NAMES = (["w", "h", "long", "short", "log_px", "v_kbps", "bpp", "i_bpp", "p_bpp", "i_over_p", "profile", "has_b", "frac_I", "frac_P", "frac_B",
                  "gop", "vfr", "avg_fps", "r_fps", "brand_mp42", "brand_isom", "enc_lavf", "has_mv", "mv_mean", "mv_std"]
                 + [f"mv_h{i}" for i in range(8)]
                 + [f"{b}_d{d}" for b in ("lo", "mid", "hi") for d in list(range(1, 10)) + ["sparsity"]]
                 + [f"benford_dev_{b}" for b in ("lo", "mid", "hi")] + ["grid_ratio", "log_lapvar"]
                 + ["a_kbps", "a_ch", "a_sr", "a_native_sr", "a_cutoff_hz", "a_cutoff_frac", "a_hf_db"])
    return np.array(vec, np.float32), NAMES
