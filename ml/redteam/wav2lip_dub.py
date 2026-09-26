"""Wav2Lip lip-sync dubbing on our own face tracks — the 'put new words in a real mouth' attack (blueprint §4.2.3).
Uses the reference model + mel conventions (Rudrabha/Wav2Lip, research/non-commercial licence); face boxes come from our
MediaPipe landmarks (satya_core.preprocess). Paste-back is the reference hard paste (that IS the residue a detector must find).
Held-out generator: used ONLY for test-time evaluation and demo tapes, never for training/validation/calibration.

  dub_clip(src_mp4, audio_wav16k, out_mp4, window=None)   window=(t0,t1) seconds -> partial dub, else the whole clip"""
from __future__ import annotations
import os, subprocess, tempfile
from functools import lru_cache
import cv2, numpy as np, torch
from scipy.io import wavfile
from scipy.signal import medfilt

from ml.redteam.wav2lip import audio as W2LA
from ml.redteam.wav2lip.wav2lip import Wav2Lip
from satya_core.preprocess import decode, track_faces, FPS

CKPT = os.path.expanduser("~/.cache/satya/weights/wav2lip/wav2lip_gan.pth")
IMG = 96


@lru_cache(maxsize=1)
def _model(device="cuda"):
    m = Wav2Lip()
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)["state_dict"]     # tensors only; no pickle code paths
    m.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})
    return m.to(device).eval()


def _boxes(pts, H, W, pad_bottom=10):
    x0, y0, x1, y1 = pts[..., 0].min(1), pts[..., 1].min(1), pts[..., 0].max(1), pts[..., 1].max(1)
    k = 5
    b = np.stack([medfilt(v, k) for v in (x0, y0, x1, y1)], 1)
    b[:, 3] += pad_bottom
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, W - 1); b[:, [1, 3]] = b[:, [1, 3]].clip(0, H - 1)
    return b.astype(int)


@torch.no_grad()
def dub_clip(src: str, wav16: np.ndarray, out: str, window: tuple[float, float] | None = None, device: str = "cuda", bs: int = 32):
    frames, own_wav, info = decode(src, max_side=960)
    T = len(frames)
    if T < 25:
        raise ValueError("clip too short")
    H, W = frames.shape[1:3]
    pts, _, ok, _, pres = track_faces(frames)
    if pres < 0.6:
        raise ValueError("no stable face")
    boxes = _boxes(pts, H, W)
    # audio must cover the clip: loop/crop the dub audio to the video length
    need = int(T / FPS * 16000) + 16000
    a = np.tile(wav16, int(np.ceil(need / max(len(wav16), 1))))[:need].astype(np.float32)
    mel = W2LA.melspectrogram(a)                                                  # (80, steps)
    step = 80.0 / FPS
    model = _model(device); out_frames = frames.copy()
    f0, f1 = (0, T) if window is None else (int(window[0] * FPS), min(T, int(window[1] * FPS)))
    idxs = list(range(f0, f1))
    for s in range(0, len(idxs), bs):
        chunk = idxs[s:s + bs]; imgs, mels, geo = [], [], []
        for i in chunk:
            x0, y0, x1, y1 = boxes[i]
            if x1 - x0 < 16 or y1 - y0 < 16:
                continue
            face = cv2.resize(frames[i][y0:y1, x0:x1], (IMG, IMG))
            m0 = int(i * step); m = mel[:, m0:m0 + 16]
            if m.shape[1] < 16: m = mel[:, -16:]
            imgs.append(face); mels.append(m.T[..., None] if False else m); geo.append((i, x0, y0, x1, y1))
        if not imgs:
            continue
        ib = np.asarray(imgs); im = ib.copy(); im[:, IMG // 2:] = 0
        ib6 = np.concatenate([im, ib], axis=3) / 255.0
        mb = np.asarray(mels)[..., None]
        x = torch.from_numpy(np.transpose(ib6, (0, 3, 1, 2))).float().to(device)
        mtens = torch.from_numpy(np.transpose(mb, (0, 3, 1, 2))).float().to(device)
        pred = model(mtens, x).cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        for p, (i, x0, y0, x1, y1) in zip(pred, geo):
            out_frames[i][y0:y1, x0:x1] = cv2.resize(p.astype(np.uint8), (x1 - x0, y1 - y0))
    # mux: dubbed video + audio (dub audio over the window, original elsewhere)
    with tempfile.TemporaryDirectory() as td:
        silent = os.path.join(td, "v.mp4")
        vw = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        for f in out_frames: vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
        base = own_wav.copy() if len(own_wav) else np.zeros(int(T / FPS * 16000), np.float32)
        if window is None:
            final = a[: int(T / FPS * 16000)]
        else:
            final = np.pad(base, (0, max(0, int(T / FPS * 16000) - len(base))))[: int(T / FPS * 16000)].copy()
            s0, s1 = int(window[0] * 16000), int(min(window[1], T / FPS) * 16000)
            final[s0:s1] = a[s0:s1]
        wp = os.path.join(td, "a.wav"); wavfile.write(wp, 16000, (np.clip(final, -1, 1) * 32767).astype(np.int16))
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", silent, "-i", wp, "-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "128k", "-shortest", out], check=True)
    return {"frames": T, "face_presence": pres, "window": window}
