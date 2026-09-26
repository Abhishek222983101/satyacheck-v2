"""The ONE preprocessing path (blueprint §4.2.4). Training and the server both call `preprocess()`; never re-implement.

decode (PyAV) -> 25 fps CFR frames + 16 kHz mono audio (first 10 s)
face: MediaPipe Face Landmarker every 2nd frame (linear interpolation between), largest-face tracking; if the full
      frame finds nothing (wide shots / small faces), retry on 3x3 overlapping 50% tiles upscaled to 512 px
align: similarity transform of 5 anchors (eye corners, nose tip, mouth corners) -> 256x256 canonical face
mouth: 96x96 around the canonical mouth centre, grayscale, centre-crop 88x88 (AV-HuBERT / lip-reading convention)
kin:   8-d per-frame lip kinematics (landmark geometry / inter-ocular distance + 4 blendshapes)
audio: 80-bin log-mel (25 ms / 10 ms), Silero-VAD speech ratio
"""
from __future__ import annotations

import itertools
import os
from dataclasses import dataclass, field
from functools import lru_cache

import av
import cv2
import numpy as np

FPS = 25
MAX_S = 10.0
SR = 16000
WEIGHTS = os.environ.get("SATYA_WEIGHTS", os.path.expanduser("~/.cache/satya/weights"))
LM_PATH = f"{WEIGHTS}/face_landmarker.task"

# 5 anchors in the 478-pt mesh: image-left eye outer, image-right eye outer, nose tip, mouth corners
ANCHORS = [33, 263, 1, 61, 291]
TEMPLATE = np.float32([[88, 100], [168, 100], [128, 140], [102, 182], [154, 182]])   # in a 256x256 canonical face
MOUTH_C = (128, 182)
BLEND = ("jawOpen", "mouthClose", "mouthFunnel", "mouthPucker")


@dataclass
class Preprocessed:
    roi: np.ndarray                 # (T,88,88) uint8 mouth crops (zeros where no face)
    faces: np.ndarray               # (16,224,224,3) uint8 aligned face crops
    kin: np.ndarray                 # (T,8) float32
    mel: np.ndarray                 # (M,80) float32 log-mel @100 fps
    wav: np.ndarray                 # (N,) float32 16 kHz mono
    face_ok: np.ndarray             # (T,) bool: real detection (not interpolated/none)
    pts_norm: np.ndarray | None = None   # (T,478,2) float16 landmarks in [0,1] image coords (reusable across re-encodes)
    blend: np.ndarray | None = None      # (T,4) float32 blendshapes
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- decode
def decode(path: str, max_s: float = MAX_S, max_side: int = 960):
    """-> frames (T,H,W,3) uint8 @25 fps CFR (nearest-time), wav float32 16 kHz mono, info dict.
    Frames are downscaled at decode time to <= max_side px on the long side (WhatsApp itself caps at ~848), which keeps
    memory and CPU bounded on 1080p/4K uploads. info['width'/'height'] describe the PROCESSED frames."""
    frames_t, frames = [], []
    with av.open(path) as c:
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"; vs.codec_context.thread_count = 2      # bounded: 8 parallel workers x default pool thrashes
        has_audio = len(c.streams.audio) > 0
        chunks = []
        rs = av.AudioResampler(format="fltp", layout="mono", rate=SR) if has_audio else None
        streams = [vs] + ([c.streams.audio[0]] if has_audio else [])
        for pkt in c.demux(*streams):
            for fr in pkt.decode():
                if isinstance(fr, av.VideoFrame):
                    t = float(fr.pts * vs.time_base) if fr.pts is not None else len(frames) / 25
                    if t <= max_s + 0.1:
                        w0, h0 = fr.width, fr.height
                        sc = min(1.0, max_side / max(w0, h0))
                        if sc < 1.0:
                            fr = fr.reformat(width=int(round(w0 * sc / 2)) * 2, height=int(round(h0 * sc / 2)) * 2, format="rgb24", interpolation="AREA")
                        frames_t.append(t); frames.append(fr.to_ndarray(format="rgb24"))
                elif rs is not None:
                    for g in rs.resample(fr):
                        chunks.append(g.to_ndarray()[0])
        if rs is not None:
            for g in rs.resample(None):
                chunks.append(g.to_ndarray()[0])
        dur_total = float(c.duration / av.time_base) if c.duration else (frames_t[-1] if frames_t else 0)
    wav = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)
    ft = np.array(frames_t)
    dur = min(max_s, ft[-1] - ft[0] + 1 / FPS if len(ft) else 0)
    n = int(round(dur * FPS))
    grid = ft[0] + np.arange(n) / FPS if len(ft) else np.zeros(0)
    idx = np.clip(np.searchsorted(ft, grid), 0, len(ft) - 1) if len(ft) else np.zeros(0, int)
    if len(ft):   # nearest, not next
        prev = np.clip(idx - 1, 0, len(ft) - 1)
        idx = np.where(np.abs(ft[prev] - grid) < np.abs(ft[idx] - grid), prev, idx)
    frames = np.stack([frames[i] for i in idx]) if n else np.zeros((0, 2, 2, 3), np.uint8)
    wav = wav[: int(max_s * SR)]
    info = {"duration_s": dur_total, "assessed_s": dur, "has_audio": has_audio and len(wav) > SR // 2,
            "width": int(frames.shape[2]) if n else 0, "height": int(frames.shape[1]) if n else 0}
    return frames, wav, info


# ----------------------------------------------------------------------------- faces
@lru_cache(maxsize=2)
def _landmarker(mode: str):
    import mediapipe as mp  # noqa: F401
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    running = vision.RunningMode.VIDEO if mode == "video" else vision.RunningMode.IMAGE
    opts = vision.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=LM_PATH), running_mode=running, num_faces=2,
        output_face_blendshapes=True, min_face_detection_confidence=0.3 if mode == "image" else 0.5,
        min_face_presence_confidence=0.3 if mode == "image" else 0.5)
    return vision.FaceLandmarker.create_from_options(opts)


def _mp_img(rgb):
    import mediapipe as mp
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


def _pack(res, w, h, ox=0, oy=0, sx=1.0, sy=1.0):
    """MediaPipe result -> list of (pts(478,2) in full-frame px, blend(4))."""
    out = []
    for k, lm in enumerate(res.face_landmarks or []):
        p = np.array([[l.x, l.y] for l in lm], np.float32) * [w, h] * [sx, sy] + [ox, oy]
        bs = {b.category_name: b.score for b in (res.face_blendshapes[k] if res.face_blendshapes else [])}
        out.append((p, np.array([bs.get(n, 0.0) for n in BLEND], np.float32)))
    return out


def _tiles(frame):
    H, W = frame.shape[:2]
    for r, c in itertools.product(range(3), range(3)):
        y0, x0 = int(r * 0.25 * H), int(c * 0.25 * W)
        t = frame[y0: y0 + H // 2, x0: x0 + W // 2]
        s = 512 / t.shape[1]
        yield cv2.resize(t, (512, int(t.shape[0] * s)), interpolation=cv2.INTER_CUBIC), x0, y0, 1 / s


def detect_face(frame, ts_ms: int, use_tiles: bool, skip_full: bool = False):
    """Returns (pts478, blend4, source) or None. Source: 'full' | 'tile'."""
    H, W = frame.shape[:2]
    if not skip_full:
        r = _pack(_landmarker("video").detect_for_video(_mp_img(frame), ts_ms), W, H)
        if r:
            return (*max(r, key=lambda z: np.ptp(z[0][:, 0]) * np.ptp(z[0][:, 1])), "full")
    if use_tiles:
        lm = _landmarker("image")
        for t, x0, y0, inv in _tiles(frame):
            r = _pack(lm.detect(_mp_img(t)), t.shape[1], t.shape[0], x0, y0, inv, inv)
            if r:
                return (*r[0], "tile")
    return None


def _kin(pts, blend):
    """8-d lip kinematics normalised by inter-ocular distance (scale/roll robust)."""
    io = np.linalg.norm(pts[33] - pts[263]) + 1e-6
    ap = np.linalg.norm(pts[13] - pts[14]) / io
    wd = np.linalg.norm(pts[61] - pts[291]) / io
    up = np.linalg.norm(pts[0] - pts[13]) / io
    lo = np.linalg.norm(pts[14] - pts[17]) / io
    return np.array([ap, wd, ap / (wd + 1e-6), up, lo, blend[0], blend[2], blend[3]], np.float32)


def track_faces(frames: np.ndarray, stride: int = 2):
    """Landmarks on every `stride`-th frame, linear interpolation between. Full-frame first; tiles if it never hits.
    Returns pts, blend, ok(per frame, detection frames only), used_tiles, presence (= hits / attempts)."""
    T = len(frames)
    pts = np.full((T, 478, 2), np.nan, np.float32); bl = np.zeros((T, 4), np.float32); ok = np.zeros(T, bool)
    _landmarker.cache_clear()          # fresh VIDEO-mode timestamp clock per clip
    use_tiles, attempts, hits, tile_attempts = False, 0, 0, 0
    for i in range(0, T, stride):
        d = detect_face(frames[i], int(i * 1000 / FPS), use_tiles, skip_full=use_tiles)
        if d is None and not use_tiles and hits == 0 and attempts >= 3:
            use_tiles = True           # nothing on the full frame in the first ~0.25 s: small/wide-shot face -> tiles
            d = detect_face(frames[i], int(i * 1000 / FPS) + 1, True, skip_full=True)
        attempts += 1
        if use_tiles:
            tile_attempts += 1
        if d is not None:
            pts[i], bl[i], _ = d; ok[i] = True; hits += 1
        elif use_tiles and hits == 0 and tile_attempts >= 6:
            break                      # faceless clip (e.g. wildlife): stop burning CPU on tiles
    have = np.nonzero(ok)[0]
    if len(have) >= 2:
        for k in range(478):
            for c in range(2):
                pts[:, k, c] = np.interp(np.arange(T), have, pts[have, k, c])
        for c in range(4):
            bl[:, c] = np.interp(np.arange(T), have, bl[have, c])
    elif len(have) == 1:
        pts[:] = pts[have[0]]; bl[:] = bl[have[0]]
    return pts, bl, ok, use_tiles, (hits / attempts if attempts else 0.0)


def align_and_crop(frames, pts):
    """Per-frame similarity alignment -> mouth ROI (T,88,88) gray + aligned faces (T,256,256,3, lazily sampled)."""
    T = len(frames)
    roi = np.zeros((T, 88, 88), np.uint8)
    # 3-tap temporal smoothing of the anchors (kills detector jitter, keeps articulation which lives in the ROI pixels)
    a = pts[:, ANCHORS]
    a = np.stack([np.convolve(np.pad(a[:, k, c], 1, mode="edge"), np.ones(3) / 3, "valid") for k in range(5) for c in range(2)], 1).reshape(T, 5, 2)
    mats = []
    for t in range(T):
        if np.isnan(a[t]).any():
            mats.append(None); continue
        M, _ = cv2.estimateAffinePartial2D(a[t].astype(np.float32), TEMPLATE, method=cv2.LMEDS)
        mats.append(M)
        if M is None:
            continue
        g = cv2.cvtColor(cv2.warpAffine(frames[t], M, (256, 256), flags=cv2.INTER_LINEAR), cv2.COLOR_RGB2GRAY)
        cx, cy = MOUTH_C
        roi[t] = g[cy - 44: cy + 44, cx - 44: cx + 44]
    return roi, mats


def face_crops(frames, mats, n=16):
    T = len(frames)
    idx = np.linspace(0, T - 1, n).round().astype(int)
    out = np.zeros((n, 224, 224, 3), np.uint8)
    for j, t in enumerate(idx):
        if mats[t] is not None:
            out[j] = cv2.warpAffine(frames[t], mats[t], (256, 256))[16:240, 16:240]
    return out


# ----------------------------------------------------------------------------- audio
@lru_cache(maxsize=1)
def _mel_fb():
    import torchaudio
    return torchaudio.transforms.MelSpectrogram(SR, n_fft=512, win_length=400, hop_length=160, n_mels=80, f_min=20, f_max=7600)


def log_mel(wav: np.ndarray) -> np.ndarray:
    import torch
    if len(wav) < 400:
        return np.zeros((0, 80), np.float32)
    m = _mel_fb()(torch.from_numpy(wav))
    return torch.log(m + 1e-6).T.numpy().astype(np.float32)


@lru_cache(maxsize=1)
def _vad():
    from silero_vad import load_silero_vad
    return load_silero_vad(onnx=True)


def speech_ratio(wav: np.ndarray) -> float:
    import torch
    from silero_vad import get_speech_timestamps
    if len(wav) < SR // 2:
        return 0.0
    ts = get_speech_timestamps(torch.from_numpy(wav), _vad(), sampling_rate=SR)
    return float(sum(t["end"] - t["start"] for t in ts) / len(wav))


# ----------------------------------------------------------------------------- entry point
def preprocess(path: str, on_stage=None, lm: dict | None = None, vad: bool = True) -> Preprocessed:
    """lm: reuse landmarks from another encode of the SAME clip: {'pts_norm','blend','ok','tiles','presence'}.
    Geometry does not change under re-encoding, so degraded copies skip the (slow) face tracking. Never use `lm`
    for a clip whose frame count/timing differs."""
    import time
    tm, t0 = {}, time.time()
    def stage(name):
        nonlocal t0
        tm[name] = round((time.time() - t0) * 1000); t0 = time.time()
        if on_stage: on_stage(name, tm[name])
    frames, wav, info = decode(path); stage("decode")
    T = len(frames)
    if T < FPS * 2:
        info["too_short"] = True
    if T and lm is not None:
        H, W = frames.shape[1:3]
        pts = (lm["pts_norm"][:T].astype(np.float32) * [W, H]).astype(np.float32)
        if len(pts) < T:   # re-encode one frame shorter/longer: pad by holding the last track
            pts = np.concatenate([pts, np.repeat(pts[-1:], T - len(pts), 0)])
        bl, ok, tiles, presence = lm["blend"][:T], lm["ok"][:T], lm["tiles"], lm["presence"]
        if len(bl) < T:
            bl = np.concatenate([bl, np.repeat(bl[-1:], T - len(bl), 0)]); ok = np.concatenate([ok, np.repeat(ok[-1:], T - len(ok))])
    elif T:
        pts, bl, ok, tiles, presence = track_faces(frames)
    else:
        pts, bl, ok, tiles, presence = np.zeros((0, 478, 2), np.float32), np.zeros((0, 4), np.float32), np.zeros(0, bool), False, 0.0
    stage("landmarks")
    roi, mats = align_and_crop(frames, pts) if T else (np.zeros((0, 88, 88), np.uint8), [])
    faces = face_crops(frames, mats) if T else np.zeros((16, 224, 224, 3), np.uint8)
    kin = np.stack([_kin(pts[t], bl[t]) if not np.isnan(pts[t]).any() else np.zeros(8, np.float32) for t in range(T)]) if T else np.zeros((0, 8), np.float32)
    stage("align")
    mel = log_mel(wav); sr = speech_ratio(wav) if (vad and info["has_audio"]) else (0.0 if not info["has_audio"] else -1.0); stage("audio")
    meta = {**info, "fps": FPS, "frames": T, "face_presence": float(presence),
            "face_detected_via_tiles": bool(tiles), "speech_ratio": sr, "timings_ms": tm,
            "no_face": presence < 0.4, "no_speech": (sr < 0.15) if sr >= 0 else False}
    H_, W_ = (frames.shape[1:3] if T else (1, 1))
    pn = (pts / [W_, H_]).astype(np.float16) if T else None
    return Preprocessed(roi=roi, faces=faces, kin=kin, mel=mel, wav=wav, face_ok=ok, pts_norm=pn, blend=bl, meta=meta)
