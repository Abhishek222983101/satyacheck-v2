"""P6.1 — jointly-labelled multi-branch samples for the fusion stage (Nyaya).

For every held-out clip (val / cal / test splits — branch models never trained on them) and forwarding generation g:
  real            untouched                                     label 0
  sync_shift      audio delayed/advanced 4..15 frames            label 1   (M1's family)
  sync_swap       speech of another clip                         label 1
  sync_partial    1.5-3 s window shifted/swapped                 label 1
  splice_replace  same-length span replaced (cut-and-paste)      label 1   (M2's family)
  face_blend      lips/lower-face self-blend on the face crops   label 1   (M4's family)
Fairness rules (so no branch can cheat on plumbing):
  * every audio condition, INCLUDING real, is re-encoded by the same in-memory AAC chain (g hops) after the attack;
  * every face-crop set, real or fake, gets the same light channel degradation;
  * attacked audio keeps the clip length, so it still lines up with the video.
Everything runs through satya_core.pipeline (the runtime path). Usage:
  python -m ml.train.nyaya_data --split val --clips 150      -> data/manifests/nyaya_val.parquet"""
from __future__ import annotations

import argparse, hashlib, json, os, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

CACHE, GENS, BANK = Path("data/cache_v"), Path("data/gens_video"), Path("data/face_bank")
CONDS = ["real", "sync_shift", "sync_swap", "sync_partial", "splice_replace", "face_blend"]
FAKE_CONDS = CONDS[1:]


def gen_faces(path: str, lm: np.ndarray, n: int = 8):
    """Aligned 224 crops + crop-space landmarks from a REAL encoded video (gen-g ladder file) using gen-0 landmarks."""
    from satya_core.preprocess import decode, ANCHORS, TEMPLATE
    frames, _, _ = decode(path, max_s=10.0, max_side=960)
    T = min(len(frames), len(lm)); H, W = frames.shape[1:3]
    idx = np.linspace(0, T - 1, n).round().astype(int); faces, lmk = [], []
    for t in idx:
        pts = lm[t].astype(np.float32) * [W, H]
        M, _ = cv2.estimateAffinePartial2D(pts[ANCHORS], TEMPLATE, method=cv2.LMEDS)
        if M is None:
            faces.append(np.zeros((224, 224, 3), np.uint8)); lmk.append(np.zeros((478, 2), np.float32)); continue
        faces.append(cv2.warpAffine(frames[t], M, (256, 256))[16:240, 16:240]); lmk.append((pts @ M[:, :2].T + M[:, 2]) - 16)
    return np.stack(faces), np.stack(lmk).astype(np.float32)


def source_wav(path: str) -> np.ndarray:
    from satya_core.preprocess import decode
    import av
    xs = []
    with av.open(path) as c:
        s = c.streams.audio[0]; rs = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        for fr in c.decode(s):
            for g in rs.resample(fr): xs.append(g.to_ndarray()[0])
        for g in rs.resample(None): xs.append(g.to_ndarray()[0])
    return np.concatenate(xs).astype(np.float32)[: 10 * 16000]


def make_sample(models, row, g, cond, rng, tab, other_wavs):
    """One (clip, generation, condition) -> feature dict. Returns None if an input is unavailable."""
    from ml.forwardchain.aac_mem import aac_chain
    from ml.forwardchain.emulate import load_priors
    from ml.redteam.splice import replace_span, energy_valleys
    from ml.data.sbi import make_fake, channel
    from satya_core.pipeline import analyze_arrays
    from satya_core.preprocess import log_mel
    from types import SimpleNamespace
    cid = row["clip_id"]; c = CACHE / cid
    z = np.load(c / f"g{g}.npz"); roi = z["roi"]; kin = np.load(c / "kin.npy").astype(np.float32)
    meta = json.loads((c / "meta.json").read_text()); T = min(len(roi), len(kin))
    wav0 = source_wav(row["path"]); n = len(wav0)
    if n < 3 * 16000 or T < 50:
        return None
    x = wav0.copy(); truth = {}
    # ---- attack on the clean (gen-0) audio
    if cond == "sync_shift":
        s = int(rng.choice([v for v in range(-15, 16) if abs(v) >= 4])); k = int(s * 640)
        x = np.roll(x, k)
        if k > 0: x[:k] = 0
        else: x[k:] = 0
    elif cond == "sync_swap" or cond == "sync_partial":
        o = other_wavs[int(rng.integers(len(other_wavs)))]; o = np.tile(o, int(np.ceil(n / len(o))))[:n]
        if cond == "sync_swap":
            x = o.copy()
        else:
            L = int(rng.uniform(1.5, 3.0) * 16000); a0 = int(rng.integers(0, max(1, n - L)))
            if rng.random() < 0.6: x[a0:a0 + L] = np.roll(x, int(rng.choice([-1, 1]) * rng.integers(5, 16) * 640))[a0:a0 + L]
            else: x[a0:a0 + L] = o[a0:a0 + L]
    elif cond == "splice_replace":
        v = energy_valleys(x, 16000); v = v[(v > 0.4) & (v < n / 16000 - 0.4)]
        if len(v) < 3: return None
        for _ in range(8):
            i = int(rng.integers(0, len(v) - 1)); j = next((j for j in range(i + 1, len(v)) if 0.25 <= v[j] - v[i] <= 0.9), None)
            if j is not None: break
        else: return None
        o = other_wavs[int(rng.integers(len(other_wavs)))] if rng.random() < 0.5 else None
        x = replace_span(x, 16000, (float(v[i]), float(v[j])), other=o, src_start=float(rng.uniform(0, max(0.1, n / 16000 - 1.5))) if o is None else None).audio
    # ---- same channel for EVERY condition: g AAC hops after the attack
    x = aac_chain(x, g, rng, load_priors()) if g > 0 else x
    mel = log_mel(x)
    # ---- faces (real gen-g frames), same light degradation for all
    src = row["path"] if g == 0 else str(GENS / cid / f"gen_{g}.mp4")
    faces, lmk = gen_faces(src, np.load(c / "lm.npy"))
    st = 0.05 + 0.05 * g
    if cond == "face_blend":
        region = str(rng.choice(["lips", "lower", "face"], p=[0.5, 0.3, 0.2]))
        faces = np.stack([make_fake(f, l, rng, region)[0] if f.any() else f for f, l in zip(faces, lmk)])
    faces = np.stack([channel(f, rng, st) if f.any() else f for f in faces])
    # ---- codec features row (from the ForwardAge table) -> M3 input
    r = tab.get((cid, g))
    if r is None: return None
    names, vec = r
    pre = SimpleNamespace(wav=x, mel=mel, roi=roi, kin=kin, faces=faces,
                          meta={"face_presence": meta["face_presence"], "speech_ratio": meta["speech_ratio"], "has_audio": True, "assessed_s": meta["assessed_s"],
                                "no_face": meta["face_presence"] < 0.4, "no_speech": meta["speech_ratio"] < 0.15})
    b = analyze_arrays(models, pre, vec, names, wav=x, mel=mel, roi=roi, kin=kin, faces=faces)
    o = dict(clip_id=cid, gen=g, cond=cond, label=int(cond != "real"), lang=row["lang"], speaker=row["speaker"],
             speech_ratio=float(meta["speech_ratio"]), face_presence=float(meta["face_presence"]),
             m1_ok=int(b.m1["status"] == "ok"), m2_ok=int(b.m2["status"] == "ok"), m4_ok=int(b.m4["status"] == "ok"),
             gen_hat=b.m3["gen_hat"], ms_total=sum(b.timings_ms.values()))
    if b.m1["status"] == "ok": o.update(m1_score=b.m1["clip_score"], m1_nosync=b.m1["p_nosync"], m1_offset=b.m1["offset_frames"], m1_nspans=len(b.m1["spans"]))
    if b.m2["status"] == "ok": o.update(m2_splice=b.m2["p_spliced"], m2_synth=b.m2["p_synthetic"], m2_maxb=b.m2["max_boundary"], m2_nb=len(b.m2["boundaries"]))
    if b.m4["status"] == "ok": o.update(m4_score=b.m4["clip_score"])
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True); ap.add_argument("--clips", type=int, default=150); ap.add_argument("--fakes-per", type=int, default=2)
    ap.add_argument("--gens", default="0,1,2,3,4,5"); ap.add_argument("--device", default="cuda"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--part", default="0/1", help="i/n: process shard i of n (parallel workers)")
    a = ap.parse_args(); pi, pn = map(int, a.part.split("/"))
    from satya_core.pipeline import Models
    models = Models(a.device)
    man = pd.read_parquet("data/manifests/video_real.parquet"); man = man[man["split"] == a.split]
    tabdf = pd.read_parquet("data/manifests/codec_feats.parquet"); feat = [c for c in tabdf.columns if c not in ("clip_id", "gen", "split", "lang", "error")]
    tab = {(r.clip_id, int(r.gen)): (feat, np.array([getattr(r, f) for f in feat], np.float32)) for r in tabdf.itertuples()}
    ok = [r for r in man.to_dict("records") if (CACHE / r["clip_id"] / "meta.json").exists() and (CACHE / r["clip_id"] / "g5.npz").exists()]
    rng0 = np.random.default_rng(a.seed); rng0.shuffle(ok); ok = ok[: a.clips]
    other = [source_wav(r["path"]) for r in ok[:40]]
    gens = [int(g) for g in a.gens.split(",")]; rows = []; t0 = time.time()
    mine = ok[pi::pn]
    for i, r in enumerate(mine):
        for g in gens:
            rng = np.random.default_rng([a.seed, int(hashlib.md5(r["clip_id"].encode()).hexdigest()[:8], 16), g])
            conds = ["real"] + list(rng.choice(FAKE_CONDS, size=a.fakes_per, replace=False))
            for cond in conds:
                try:
                    o = make_sample(models, r, g, cond, rng, tab, other)
                except Exception as e:
                    print("skip", r["clip_id"][:20], g, cond, repr(e)[:80], flush=True); o = None
                if o: rows.append(o)
        if (i + 1) % 10 == 0: print(f"{i+1}/{len(mine)} clips, {len(rows)} samples, {time.time()-t0:.0f}s", flush=True)
    out = Path(f"data/manifests/nyaya_{a.split}_{pi}of{pn}.parquet"); pd.DataFrame(rows).to_parquet(out, index=False); print("wrote", out, len(rows))


if __name__ == "__main__":
    main()
