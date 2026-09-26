"""Audio half of ForwardChain: k generations of AAC re-encode (bitrate / sample-rate / mono drawn per hop from the
same priors as the video emulator). gen_k is produced from gen_{k-1}, so loss accumulates like real forwards.
Outputs .m4a next to a manifest column; `load_audio` decodes any of them to 16 kHz float32."""
from __future__ import annotations
import hashlib, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import av, numpy as np, pandas as pd
from scipy.signal import resample_poly
from ml.forwardchain.emulate import load_priors

SR = 16000

def load_audio(path: str | Path, sr: int = SR) -> np.ndarray:
    with av.open(str(path)) as c:
        s = c.streams.audio[0]; rs = av.AudioResampler(format="fltp", layout="mono", rate=sr)
        out = []
        for fr in c.decode(s):
            for f in rs.resample(fr): out.append(f.to_ndarray()[0])
        for f in rs.resample(None): out.append(f.to_ndarray()[0])
    return np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)

def _seed(clip_id: str, k: int) -> np.random.Generator:
    return np.random.default_rng([int(hashlib.md5(clip_id.encode()).hexdigest()[:8], 16), k])

def chain_one(args):
    clip_id, wav, out_dir, gens = args
    P = load_priors(); cur = Path(wav); out = Path(out_dir); out.mkdir(parents=True, exist_ok=True); ok = []
    for k in range(1, gens + 1):
        dst = out / f"{clip_id}_g{k}.m4a"
        if not dst.exists():
            r = _seed(clip_id, k)
            kb = int(P["aac_kbps"]["values"][r.choice(len(P["aac_kbps"]["values"]), p=P["aac_kbps"]["p"])])
            sr = int(P["audio_sr"]["values"][r.choice(len(P["audio_sr"]["values"]), p=P["audio_sr"]["p"])])
            ch = 1 if r.random() < P["p_mono"] else 2
            cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(cur), "-vn", "-ac", str(ch), "-ar", str(sr), "-c:a", "aac", "-b:a", f"{kb}k", str(dst)]
            if subprocess.run(cmd).returncode != 0: return clip_id, ok
        ok.append(str(dst)); cur = dst
    return clip_id, ok

def ladder(manifest: str, out_dir="data/audio_gens", gens=5, workers=10):
    df = pd.read_parquet(manifest)
    jobs = [(r.clip_id, r.path, out_dir, gens) for r in df.itertuples()]
    done = 0
    with ProcessPoolExecutor(workers) as ex:
        for cid, ok in ex.map(chain_one, jobs, chunksize=16):
            done += len(ok) == gens
            if done % 500 == 0: print(f"{done}/{len(jobs)}", flush=True)
    print(f"ladder complete for {done}/{len(jobs)} clips")

if __name__ == "__main__":
    ladder(sys.argv[1] if len(sys.argv) > 1 else "data/manifests/audio_split.parquet")
