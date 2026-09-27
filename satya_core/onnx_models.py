"""ONNX Runtime backend for the pipeline (P8.2). Same interface as satya_core.pipeline.Models:
m1()/m2()/m2_thr()/m4()/m3() return the exact objects run_m1..run_m3 expect, so analyze_path
works unchanged. CPU-only (ORT CPUExecutionProvider), sessions created once, hash-checked
against models/registry.json (blueprint §4.12: no silent model swaps).

Fixed-window adapters (exports are static-window, see ml/export/):
  M1: pad roi/mel/kin to the 10 s window (250 frames), slice outputs back to valid T.
  M2: pad wav to 160000 with n_valid; the exported graph masks internally; outputs sliced.
  M4: prep (ImageNet normalise) done here in numpy; batch-dynamic, no padding.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
M1_T, MEL_T, M2_WIN = 250, 1000, 160000
_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)


def _sha16(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _sess(path: str, threads: int = 4) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


class OnnxModels:
    """Drop-in for satya_core.pipeline.Models with device='cpu'."""

    def __init__(self, registry: str = "models/registry.json"):
        self.dev = "cpu"
        self.versions: dict = {}
        reg = json.loads(Path(registry).read_text())
        self.reg = reg
        self._m: dict = {}

    def _get(self, key: str, build):
        if key not in self._m:
            self._m[key] = build()
        return self._m[key]

    def _check(self, name: str, path: str) -> None:
        """Refuse to run a swapped model: sha must match the registry entry."""
        want = (self.reg.get("artifacts", {}).get(name) or {}).get("sha256_int8" if path.endswith(".int8.onnx") else "sha256")
        if want and not want.startswith(_sha16(path)):
            raise RuntimeError(f"registry hash mismatch for {name} ({path}) — refusing to load")

    def _sess(self, name: str, path: str, threads: int = 4) -> ort.InferenceSession:
        self._check(name, path)
        return self._get(f"s_{name}", lambda: _sess(path, threads))

    def _artifact(self, name: str) -> tuple[str, str]:
        """(path, variant) honouring the registry's serving decision (int8 vs fp32)."""
        a = self.reg["artifacts"][name]
        if self.reg.get("serving", {}).get(name) == "int8" and a.get("int8_path"):
            return a["int8_path"], "int8"
        return a["onnx_path"], "fp32"

    # ---- M1 -------------------------------------------------------------
    def m1(self):
        path, variant = self._artifact("syncformer")
        s = self._sess("syncformer", path)
        self.versions["syncformer"] = f"onnx {variant}"

        def call(roi, mel, kin):
            roi, mel, kin = _np(roi), _np(mel), _np(kin)
            B, T = roi.shape[0], roi.shape[1]
            rp = np.zeros((B, M1_T, *roi.shape[2:]), roi.dtype); rp[:, :T] = roi
            mp = np.zeros((B, MEL_T, mel.shape[2]), np.float32); mp[:, :mel.shape[1]] = mel
            kp = np.zeros((B, M1_T, kin.shape[2]), np.float32); kp[:, :T] = kin
            delta, fake = s.run(None, {"roi": rp, "mel": mp, "kin": kp})
            return {"delta": torch.from_numpy(delta[:, :T])}

        return call

    # ---- M2 -------------------------------------------------------------
    def m2(self):
        path, variant = self._artifact("spliceloc")
        s = self._sess("spliceloc", path)
        self.versions["spliceloc"] = f"onnx {variant}"

        def call(wav, n_valid, valid=None):
            x = _np(wav)
            B, L = x.shape
            Lc = min(L, M2_WIN)
            xp = np.zeros((B, M2_WIN), np.float32); xp[:, :Lc] = x[:, :Lc]
            nv = np.minimum(_np(n_valid), M2_WIN)
            b, f, u = s.run(None, {"wav": xp, "n_valid": nv})
            T_valid = (nv[0] - 400) // 320 + 1
            return {"boundary": torch.from_numpy(b[:, :T_valid]), "frame": torch.from_numpy(f[:, :T_valid]),
                    "utt": torch.from_numpy(u)}

        return call

    def m2_thr(self) -> float:
        if "m2_thr" not in self._m:
            ck = torch.load(ROOT / "ml/runs/spliceloc_v4/best.pt", map_location="cpu", weights_only=True)
            self._m["m2_thr"] = float(ck["metrics"].get("boundary_thr", 0.5))
            self.versions["spliceloc"] = f"onnx ep{ck['epoch']}"
        return self._m["m2_thr"]

    # ---- M4 -------------------------------------------------------------
    def m4(self):
        path, variant = self._artifact("facetrace")
        s = self._sess("facetrace", path)
        self.versions["facetrace"] = f"onnx {variant}"

        def call(x):
            xf = _np(x).astype(np.float32) / 255.0
            if xf.shape[1] == 3:  # (B,3,H,W) already channel-first
                xf = (xf - _MEAN.reshape(1, 3, 1, 1)) / _STD.reshape(1, 3, 1, 1)
                return torch.from_numpy(s.run(None, {"x": xf.astype(np.float32)})[0])
            xf = (xf - _MEAN) / _STD  # (B,H,W,3)
            xt = xf.transpose(0, 3, 1, 2).astype(np.float32)
            return torch.from_numpy(s.run(None, {"x": xt})[0])

        return call

    # ---- M3 (LightGBM on CPU, unchanged) --------------------------------
    def m3(self):
        def b():
            import lightgbm as lgb
            d = ROOT / "models"
            feats = json.loads((d / "forwardage_features.json").read_text())["features"]
            boosters = [lgb.Booster(model_file=str(d / f"forwardage_gt{k}.txt")) for k in range(5)]
            self.versions["forwardage"] = "lgbm5"
            return feats, boosters
        return self._get("m3", b)
