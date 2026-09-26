"""P2.3 (route: SyncNet) — teacher features for every cached clip, computed on the CLEAN original (gen 0).
The student later sees degraded (Gen 0-5) inputs but regresses these clean-teacher embeddings: compression-robust
distillation. Also stores the teacher's own offset/confidence (zero-shot baseline numbers).
Output: data/cache_v/<clip>/teacher.npz  {vf (n,1024) fp16, af (n,1024) fp16, offset, conf}   n = windows (T-4)
Usage: python -m ml.teachers.syncnet.extract [--device cuda] [--workers 3]"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd

CACHE = Path("data/cache_v")

def run(args):
    dev, rows = args
    import torch; torch.set_num_threads(2)
    from satya_core.preprocess import decode
    from ml.teachers.syncnet.teacher import SyncNetTeacher, syncnet_crops
    T = SyncNetTeacher(dev); res = {}
    for r in rows:
        out = CACHE / r["clip_id"] / "teacher.npz"
        if out.exists() or not (CACHE / r["clip_id"] / "meta.json").exists():
            continue
        try:
            frames, wav, _ = decode(r["path"])
            lm = np.load(CACHE / r["clip_id"] / "lm.npy").astype(np.float32)
            H, W = frames.shape[1:3]; n = min(len(frames), len(lm))
            faces = syncnet_crops(frames[:n], lm[:n] * [W, H])
            vf, af = T.embed(faces, wav)
            if len(vf) < 10: continue
            _, off, conf = T.profile(vf, af)
            np.savez(out, vf=vf.astype(np.float16), af=af.astype(np.float16), offset=off, conf=conf)
            res[r["clip_id"]] = (off, conf)
        except Exception as e:
            print("ERR", r["clip_id"], repr(e)[:100], flush=True)
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda"); ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    df = pd.read_parquet("data/manifests/video_real.parquet"); rows = df.to_dict("records")
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    chunks = [rows[i::a.workers] for i in range(a.workers)]
    with mp.Pool(a.workers) as pool:
        allr = {}
        for d in pool.map(run, [(a.device, c) for c in chunks]): allr.update(d)
    offs = np.array([v[0] for v in allr.values()]); conf = np.array([v[1] for v in allr.values()])
    if len(offs): print(f"teacher on {len(offs)} clips: |offset|<=2 frames: {np.mean(np.abs(offs) <= 2):.2f}, median conf {np.median(conf):.2f}")
    print("TEACHER_DONE")
