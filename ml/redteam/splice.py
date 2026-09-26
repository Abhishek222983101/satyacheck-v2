"""RedTeam splice factory (blueprint §4.2.3) — the cut-and-join attack, with exact labels.

Two entry points:
  * `random_splice(x, sr, rng)`      — for M2 training. Cut points snap to energy valleys (no ASR needed, fast).
  * `splice_spans(x, sr, ...)`       — for the live demo. Caller supplies exact spans (e.g. Whisper word times).

Every op is an `assemble()` of pieces joined with a short equal-power crossfade; boundaries are the crossfade
centres in the OUTPUT timeline, so labels are exact by construction.

Label conventions (output timeline, 50 fps = 20 ms frames, matching XLS-R's frame rate):
  frame_spoof[k]    = 1 if frame k overlaps inserted/duplicated/swapped material            (frame head)
  frame_boundary[k] = 1 if frame k is within ±2 frames of a join                            (boundary head)
  utterance label   = 1 for any spliced output (the clip is not one continuous take)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FRAME_S = 0.02  # 50 fps


@dataclass
class SpliceResult:
    audio: np.ndarray
    sr: int
    op: str
    boundaries: list[float] = field(default_factory=list)          # seconds, output timeline
    spoof_spans: list[tuple[float, float]] = field(default_factory=list)  # inserted material, output timeline
    removed: list[tuple[float, float]] = field(default_factory=list)      # deleted material, ORIGINAL timeline
    frame_spoof: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int8))
    frame_boundary: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int8))

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sr


# ---------------------------------------------------------------- low-level helpers
def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))


def _snap_zero(x: np.ndarray, i: int, sr: int, radius_ms: float = 5.0) -> int:
    """Move cut index to the nearest sign change within ±radius (avoids clicks)."""
    r = int(sr * radius_ms / 1000)
    lo, hi = max(1, i - r), min(len(x) - 1, i + r)
    if hi <= lo:
        return i
    seg = np.signbit(x[lo - 1:hi])
    zc = np.nonzero(seg[1:] != seg[:-1])[0]
    if len(zc) == 0:
        return i
    return int(lo + zc[np.argmin(np.abs((lo + zc) - i))])


def _match_level(piece: np.ndarray, ctx: np.ndarray) -> np.ndarray:
    """Scale an inserted piece to the loudness of its surroundings (bounded 0.2-5x; a careful forger normalises loudness, so training fakes should too)."""
    if len(piece) == 0 or len(ctx) == 0:
        return piece
    g = np.clip(_rms(ctx) / _rms(piece), 0.2, 5.0)
    return (piece * g).astype(piece.dtype)


def assemble(pieces: list[tuple[np.ndarray, bool]], sr: int, xf_ms: float = 8.0):
    """Join pieces with equal-power crossfades. Returns (audio, boundaries_s, spoof_spans_s)."""
    xf = max(0, int(sr * xf_ms / 1000))
    out = pieces[0][0].astype(np.float32).copy()
    bnds: list[float] = []
    for audio, _ in pieces[1:]:
        audio = audio.astype(np.float32)
        n = min(xf, len(out), len(audio))
        if n > 1:
            t = np.linspace(0, np.pi / 2, n, dtype=np.float32)
            mix = out[-n:] * np.cos(t) + audio[:n] * np.sin(t)
            join = len(out) - n / 2
            out = np.concatenate([out[:-n], mix, audio[n:]])
        else:
            join = len(out)
            out = np.concatenate([out, audio])
        bnds.append(join / sr)
    # spoof spans: from join before a spoof piece to the join after it (or end of audio)
    spans = []
    for i, (_, is_spoof) in enumerate(pieces):
        if not is_spoof:
            continue
        t0 = 0.0 if i == 0 else bnds[i - 1]
        t1 = len(out) / sr if i == len(pieces) - 1 else bnds[i]
        spans.append((t0, t1))
    return out, bnds, spans


def _frame_labels(n_samples: int, sr: int, bnds, spans):
    n = int(np.ceil(n_samples / sr / FRAME_S))
    spoof = np.zeros(n, np.int8)
    bound = np.zeros(n, np.int8)
    for t0, t1 in spans:
        spoof[int(t0 / FRAME_S): int(np.ceil(t1 / FRAME_S))] = 1
    for b in bnds:
        k = int(b / FRAME_S)
        bound[max(0, k - 2): k + 3] = 1
    return spoof, bound


def _finish(x_out, sr, op, bnds, spans, removed=()):
    fs, fb = _frame_labels(len(x_out), sr, bnds, spans)
    return SpliceResult(x_out, sr, op, list(bnds), list(spans), list(removed), fs, fb)


# ---------------------------------------------------------------- span-based ops (live demo)
def splice_spans(x: np.ndarray, sr: int, op: str, span: tuple[float, float],
                 span2: tuple[float, float] | None = None, xf_ms: float = 8.0,
                 other: np.ndarray | None = None) -> SpliceResult:
    """op ∈ delete | duplicate | swap | insert.  Spans are in seconds on the ORIGINAL timeline.
    delete    remove span
    duplicate repeat span immediately after itself
    swap      exchange span with span2 (span2 must start at/after span end)
    insert    put `other` (another take, same sr) at span[0]
    """
    n = len(x)
    a0 = _snap_zero(x, int(span[0] * sr), sr)
    a1 = _snap_zero(x, int(span[1] * sr), sr)
    if not 0 < a0 < a1 <= n:
        raise ValueError("span outside clip")
    if op == "delete":
        out, b, sp = assemble([(x[:a0], False), (x[a1:], False)], sr, xf_ms)
        return _finish(out, sr, op, b, sp, removed=[(a0 / sr, a1 / sr)])
    if op == "duplicate":
        seg = x[a0:a1]
        out, b, sp = assemble([(x[:a1], False), (seg, True), (x[a1:], False)], sr, xf_ms)
        return _finish(out, sr, op, b, sp)
    if op == "swap":
        if span2 is None:
            raise ValueError("swap needs span2")
        b0 = _snap_zero(x, int(span2[0] * sr), sr)
        b1 = _snap_zero(x, int(span2[1] * sr), sr)
        if not a1 <= b0 < b1 <= n:
            raise ValueError("span2 must follow span")
        A, mid, B = x[a0:a1], x[a1:b0], x[b0:b1]
        pieces = [(x[:a0], False), (B, True), (mid, False), (A, True), (x[b1:], False)]
        pieces = [(p, s) for p, s in pieces if len(p)]
        out, b, sp = assemble(pieces, sr, xf_ms)
        return _finish(out, sr, op, b, sp)
    if op == "insert":
        if other is None:
            raise ValueError("insert needs `other`")
        ins = _match_level(other, np.concatenate([x[max(0, a0 - sr // 2):a0], x[a0:a0 + sr // 2]]))
        out, b, sp = assemble([(x[:a0], False), (ins, True), (x[a0:], False)], sr, xf_ms)
        return _finish(out, sr, op, b, sp)
    raise ValueError(f"unknown op {op}")


# ---------------------------------------------------------------- random ops (training)
def energy_valleys(x: np.ndarray, sr: int, hop_ms: float = 10.0, min_gap_ms: float = 150.0) -> np.ndarray:
    """Candidate cut times (s): local minima of smoothed short-time energy, in the lower energy half."""
    hop = int(sr * hop_ms / 1000)
    n = len(x) // hop
    if n < 8:
        return np.array([])
    e = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    k = np.ones(5) / 5
    es = np.convolve(e, k, mode="same")
    thr = np.percentile(es, 50)
    idx = [i for i in range(2, n - 2) if es[i] <= es[i - 1] and es[i] <= es[i + 1] and es[i] <= thr]
    keep, last = [], -10**9
    for i in idx:
        if (i - last) * hop_ms >= min_gap_ms:
            keep.append(i)
            last = i
    return np.array(keep) * hop_ms / 1000


def random_splice(x: np.ndarray, sr: int, rng: np.random.Generator, op: str | None = None,
                  other: np.ndarray | None = None, xf_ms: float | None = None) -> SpliceResult | None:
    """One random cut-and-join at energy valleys. Returns None if the clip has too few cut candidates."""
    dur = len(x) / sr
    cuts = energy_valleys(x, sr)
    cuts = cuts[(cuts > 0.3) & (cuts < dur - 0.3)]
    if len(cuts) < 3:
        return None
    op = op or str(rng.choice(["delete", "swap", "duplicate", "insert" if other is not None else "delete"]))
    xf = float(rng.uniform(0, 15)) if xf_ms is None else xf_ms
    for _ in range(10):  # find a span of 0.2–0.8 s between two valleys
        i = int(rng.integers(0, len(cuts) - 1))
        for j in range(i + 1, len(cuts)):
            L = cuts[j] - cuts[i]
            if 0.2 <= L <= 0.8:
                span = (float(cuts[i]), float(cuts[j]))
                if op == "swap":
                    nxt = [c for c in cuts if c > span[1] + 0.05]
                    nxt2 = [c for c in nxt if 0.2 <= c - nxt[0] <= 0.8] if nxt else []
                    if not nxt2:
                        break
                    return splice_spans(x, sr, "swap", span, (float(nxt[0]), float(nxt2[0])), xf)
                if op == "insert":
                    seg = other[: int(L * sr)] if len(other) >= int(L * sr) else other
                    return splice_spans(x, sr, "insert", (span[0], span[0] + 0.01), xf_ms=xf, other=seg)
                return splice_spans(x, sr, op, span, xf_ms=xf)
    return None
