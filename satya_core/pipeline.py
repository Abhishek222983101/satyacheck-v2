"""Runtime pipeline — the four branches on one preprocessed clip (PyTorch backend; the ONNX backend swaps in at P8 with the
same function signatures and outputs). One code path for: Nyaya's training data, evaluation, and the server.

  M1 SyncFormer  -> sync curve / spans / offset / clip score          (needs face + speech)
  M2 SpliceLoc   -> splice boundaries / frame spoof / synthetic voice (needs speech)
  M4 FaceTrace   -> blending-artefact probability                     (needs face)
  M3 ForwardAge  -> generation distribution                           (always)
Branches that cannot run report status 'abstained' with a reason code (blueprint §4.8) — never a guess.
All checkpoints are loaded with weights_only=True (tensors only, no pickle code execution)."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = {
    "m1": ROOT / "ml/runs/syncformer/best.pt",
    "m2": ROOT / "ml/runs/spliceloc_v4/best.pt",
    "m4": ROOT / "ml/runs/facetrace_v2/best.pt",
    "m3": ROOT / "models",
}
NOSYNC, ZERO = 31, 15


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


class Models:
    """Lazy, cached model set. `device` applies to M1/M2/M4; M3 is LightGBM on CPU."""

    def __init__(self, device: str = "cuda", ckpt: dict | None = None):
        self.dev = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.ck = {**DEFAULT_CKPT, **(ckpt or {})}
        self._m: dict = {}
        self.versions: dict = {}

    def _get(self, k, build):
        if k not in self._m:
            self._m[k] = build()
        return self._m[k]

    def m1(self):
        def b():
            from ml.train.syncformer import SyncFormer
            ck = _load(self.ck["m1"]); m = SyncFormer(); m.load_state_dict(ck["model"]); self.versions["syncformer"] = f"ep{ck['epoch']}"
            return m.to(self.dev).eval()
        return self._get("m1", b)

    def m2(self):
        def b():
            from ml.train.spliceloc import SpliceLoc
            ck = _load(self.ck["m2"]); m = SpliceLoc(); m.load_state_dict(ck["model"])
            self.versions["spliceloc"] = f"ep{ck['epoch']}"; self._m["m2_thr"] = float(ck["metrics"].get("boundary_thr", 0.5))
            return m.to(self.dev).eval()
        return self._get("m2", b)

    def m2_thr(self):
        self.m2(); return self._m["m2_thr"]

    def m4(self):
        def b():
            from ml.train.facetrace import build
            ck = _load(self.ck["m4"]); m = build(pretrained=False); m.load_state_dict(ck["model"]); self.versions["facetrace"] = f"ep{ck['epoch']}"
            return m.to(self.dev).eval()
        return self._get("m4", b)

    def m3(self):
        def b():
            import lightgbm as lgb
            d = Path(self.ck["m3"]); feats = json.loads((d / "forwardage_features.json").read_text())["features"]
            boosters = [lgb.Booster(model_file=str(d / f"forwardage_gt{k}.txt")) for k in range(5)]
            self.versions["forwardage"] = "lgbm5"
            return feats, boosters
        return self._get("m3", b)


# ----------------------------------------------------------------------------- branches
def _spans(bad: np.ndarray, fps: float, thr=0.5, min_len=0.6):
    """Contiguous runs of `bad > thr` lasting >= min_len seconds -> [(t0, t1, mean score)]."""
    on, out, s = bad > thr, [], None
    for i, v in enumerate(list(on) + [False]):
        if v and s is None: s = i
        if not v and s is not None:
            if (i - s) / fps >= min_len: out.append((s / fps, i / fps, float(bad[s:i].mean())))
            s = None
    return out


@torch.no_grad()
def run_m1(models: Models, roi, mel, kin):
    """roi (T,88,88) uint8, mel (M,80), kin (T,8) -> dict. Sync = P(|shift|<=2 frames); natural clips carry ±2 frames."""
    T = min(len(roi), len(mel) // 4, len(kin))
    if T < 50:
        return {"status": "abstained", "reason": "too_short"}
    x = (torch.from_numpy(roi[:T])[None].to(models.dev), torch.from_numpy(mel[:4 * T].astype(np.float32))[None].to(models.dev), torch.from_numpy(kin[:T].astype(np.float32))[None].to(models.dev))
    with torch.autocast(models.dev if models.dev == "cuda" else "cpu", dtype=torch.bfloat16, enabled=models.dev == "cuda"):
        o = models.m1()(*x)
    p = F.softmax(o["delta"].float(), -1)[0].cpu().numpy()                      # (T,32)
    sp = p[:, ZERO - 2: ZERO + 3].sum(-1)
    win = 40
    ms = np.convolve(sp, np.ones(win) / win, "valid") if T >= win else np.array([sp.mean()])
    cm = p.mean(0)
    off = int(np.argmax(cm[:NOSYNC])) - ZERO
    bad = 1 - np.convolve(np.pad(sp, (2, 2), mode="edge"), np.ones(5) / 5, "valid")
    return {"status": "ok", "clip_score": float(1 - ms.min()), "sync_prob": sp.astype(np.float32), "bad": bad.astype(np.float32),
            "p_nosync": float(cm[NOSYNC]), "offset_frames": off, "offset_ms": 40 * off, "spans": _spans(bad, 25.0), "class_mean": cm}


@torch.no_grad()
def run_m2(models: Models, wav, max_s=10.0):
    """wav float32 16 kHz -> splice boundaries, frame spoof, p(synthetic), p(spliced)."""
    x = wav[: int(max_s * 16000)]
    if len(x) < 3 * 16000:
        return {"status": "abstained", "reason": "too_short"}
    t = torch.from_numpy(x.astype(np.float32))[None].to(models.dev); n = torch.tensor([t.shape[1]], device=models.dev)
    with torch.autocast(models.dev if models.dev == "cuda" else "cpu", dtype=torch.bfloat16, enabled=models.dev == "cuda"):
        o = models.m2()(t, n)
    pb = torch.sigmoid(o["boundary"].float())[0].cpu().numpy(); fs = F.softmax(o["frame"].float(), -1)[0, :, 1].cpu().numpy()
    us = torch.sigmoid(o["utt"].float())[0].cpu().numpy()
    from ml.train.spliceloc import pick_peaks
    peaks = pick_peaks(pb, models.m2_thr())
    return {"status": "ok", "p_synthetic": float(us[0]), "p_spliced": float(us[1]), "frame_spoof": fs.astype(np.float32), "boundary_prob": pb.astype(np.float32),
            "boundaries": [{"t": float(i * 0.02 + 0.0125), "strength": float(pb[i])} for i in peaks], "max_boundary": float(pb.max())}


@torch.no_grad()
def run_m4(models: Models, faces):
    """faces (N,224,224,3) uint8 RGB aligned crops (all-zero crops = no face at that frame and are skipped)."""
    keep = [i for i in range(len(faces)) if faces[i].any()]
    if len(keep) < 4:
        return {"status": "abstained", "reason": "no_face"}
    from ml.train.facetrace import prep
    x = prep(torch.from_numpy(faces[keep]).permute(0, 3, 1, 2).to(models.dev))
    with torch.autocast(models.dev if models.dev == "cuda" else "cpu", dtype=torch.bfloat16, enabled=models.dev == "cuda"):
        p = torch.sigmoid(models.m4()(x).float().squeeze(-1)).cpu().numpy()
    k = max(1, len(p) // 4)
    return {"status": "ok", "frame_probs": p.astype(np.float32), "clip_score": float(np.sort(p)[-k:].mean())}


def run_m3(models: Models, codec_vec: np.ndarray, names: list[str]):
    feats, boosters = models.m3()
    row = np.array([[float(dict(zip(names, codec_vec)).get(f, 0.0)) for f in feats]], np.float32)
    P = np.stack([b.predict(row)[0] for b in boosters])[None]
    P = np.minimum.accumulate(np.clip(P, 0, 1), axis=1); edges = np.concatenate([np.ones((1, 1)), P, np.zeros((1, 1))], 1)
    D = (edges[:, :-1] - edges[:, 1:])[0]
    return {"status": "ok", "gen_hat": float((D * np.arange(6)).sum()), "gen_dist": D.astype(np.float32)}


@dataclass
class BranchOut:
    m1: dict; m2: dict; m4: dict; m3: dict
    timings_ms: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def analyze_arrays(models: Models, pre, codec_vec, codec_names, wav=None, mel=None, roi=None, kin=None, faces=None) -> BranchOut:
    """pre: satya_core.preprocess.Preprocessed. Overrides (wav/mel/roi/kin/faces) let attack builders swap one modality
    (e.g. tampered audio -> BOTH its mel for M1 and its waveform for M2) without touching the rest."""
    from satya_core.preprocess import log_mel
    tm, t0 = {}, time.time()
    wav = pre.wav if wav is None else wav
    mel = (log_mel(wav) if (wav is not pre.wav and mel is None) else pre.mel) if mel is None else mel
    roi = pre.roi if roi is None else roi; kin = pre.kin if kin is None else kin; faces = pre.faces if faces is None else faces
    m = pre.meta; has_face = not m.get("no_face", False); has_speech = not m.get("no_speech", False) and m.get("has_audio", False)
    def timed(name, fn):
        nonlocal t0
        r = fn(); tm[name] = round((time.time() - t0) * 1000); t0 = time.time(); return r
    m1 = timed("sync", lambda: run_m1(models, roi, mel, kin) if (has_face and has_speech) else {"status": "abstained", "reason": "no_face" if not has_face else "no_speech"})
    m2 = timed("splice", lambda: run_m2(models, wav) if has_speech else {"status": "abstained", "reason": "no_speech"})
    m4 = timed("face", lambda: run_m4(models, faces) if has_face else {"status": "abstained", "reason": "no_face"})
    m3 = timed("age", lambda: run_m3(models, codec_vec, codec_names))
    return BranchOut(m1, m2, m4, m3, tm, {k: m.get(k) for k in ("face_presence", "speech_ratio", "has_audio", "assessed_s", "no_face", "no_speech")})
