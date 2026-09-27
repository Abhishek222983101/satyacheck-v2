"""SyncNet (Chung & Zisserman; MIT, joonson/syncnet_python) as (a) distillation TEACHER for SyncFormer and
(b) a zero-shot lip-sync BASELINE row in the benchmark.

Input convention (from the reference implementation): 224x224 BGR face crops at 25 fps, 0-255 float, 5-frame windows;
audio = 13-d MFCC at 100 Hz, 20 MFCC frames (0.2 s) per video window. Outputs 1024-d embeddings for each modality.
`profile()` returns the distance to audio windows at offsets -vshift..+vshift; the min over the clip's mean profile
is the AV offset, (median - min) the confidence (LSE-D / LSE-C)."""
from __future__ import annotations
import os
import numpy as np, torch
from python_speech_features import mfcc as _mfcc
from ml.teachers.syncnet.SyncNetModel import S

WEIGHTS = os.path.expanduser("~/.cache/satya/weights/syncnet_v2.model")


class SyncNetTeacher:
    def __init__(self, device: str = "cuda"):
        self.dev = device
        self.net = S(num_layers_in_fc_layers=1024)
        state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)      # tensors only: no pickle code execution
        own = self.net.state_dict()
        for k, v in state.items():
            own[k].copy_(v)
        self.net.to(device).eval()

    @staticmethod
    def mfcc_feats(wav16k: np.ndarray) -> np.ndarray:
        a = np.clip(wav16k * 32768, -32768, 32767).astype(np.int16)
        return np.stack(list(zip(*_mfcc(a, 16000)))).astype(np.float32)          # (13, ~100*sec)

    @torch.no_grad()
    def embed(self, faces_bgr: np.ndarray, wav16k: np.ndarray, batch: int = 64):
        """faces_bgr (T,224,224,3) uint8 -> vfeat (n,1024), afeat (n,1024) with n = min(T-4, audio windows)."""
        T = len(faces_bgr)
        mf = self.mfcc_feats(wav16k)
        n = min(T - 4, mf.shape[1] // 4 - 4)
        if n <= 0:
            return np.zeros((0, 1024), np.float32), np.zeros((0, 1024), np.float32)
        im = torch.from_numpy(faces_bgr.astype(np.float32)).permute(3, 0, 1, 2)[None]        # (1,3,T,H,W)
        cc = torch.from_numpy(mf)[None, None]                                                # (1,1,13,M)
        vf, af = [], []
        for i in range(0, n, batch):
            idx = range(i, min(n, i + batch))
            vb = torch.cat([im[:, :, j:j + 5] for j in idx], 0).to(self.dev)
            ab = torch.cat([cc[:, :, :, j * 4: j * 4 + 20] for j in idx], 0).to(self.dev)
            vf.append(self.net.forward_lip(vb).cpu()); af.append(self.net.forward_aud(ab).cpu())
        return torch.cat(vf).numpy(), torch.cat(af).numpy()

    @staticmethod
    def profile(vf: np.ndarray, af: np.ndarray, vshift: int = 15):
        """-> dists (n, 2*vshift+1) L2 distance of video window i to audio window i+off, offset (frames), confidence."""
        v, a = torch.from_numpy(vf), torch.from_numpy(af)
        ap = torch.nn.functional.pad(a, (0, 0, vshift, vshift))
        D = torch.stack([torch.nn.functional.pairwise_distance(v, ap[k:k + len(v)]) for k in range(2 * vshift + 1)], 1)   # col k <-> offset k-vshift
        m = D.mean(0)
        k = int(torch.argmin(m))
        return D.numpy(), k - vshift, float(torch.median(m) - m[k])


def syncnet_crops(frames_rgb: np.ndarray, pts: np.ndarray, cs: float = 0.4) -> np.ndarray:
    """Reference-implementation face crop (the aligned 224 crop gave confidence 0.4; this gives 2-8).
    Box = landmark hull; centre/size median-filtered over time (as the reference tracker does); returns (T,224,224,3) BGR."""
    import cv2
    from scipy.signal import medfilt
    x0, y0, x1, y1 = pts[..., 0].min(1), pts[..., 1].min(1), pts[..., 0].max(1), pts[..., 1].max(1)
    k = 9 if len(x0) >= 9 else 1
    cx, cy, bs = medfilt((x0 + x1) / 2, k), medfilt((y0 + y1) / 2, k), medfilt(np.maximum(y1 - y0, x1 - x0) / 2, k)
    out = np.zeros((len(frames_rgb), 224, 224, 3), np.uint8)
    H, W = frames_rgb.shape[1:3]
    for t, f in enumerate(frames_rgb):
        b = max(float(bs[t]), 8.0)
        # reference crop = window on a frame padded with grey(110); crop first, pad only the window (identical result, ~10x cheaper)
        y0, y1 = int(cy[t] - b), int(cy[t] + b * (1 + 2 * cs)); x0, x1 = int(cx[t] - b * (1 + cs)), int(cx[t] + b * (1 + cs))
        sub = f[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
        face = np.pad(sub, ((max(0, -y0), max(0, y1 - H)), (max(0, -x0), max(0, x1 - W)), (0, 0)), "constant", constant_values=110)
        out[t] = cv2.resize(face, (224, 224)) if face.size and face.shape[0] > 1 and face.shape[1] > 1 else out[t]
    return out[..., ::-1].copy()          # RGB -> BGR
