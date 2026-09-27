"""Generate the HELD-OUT lip-sync attack set: Wav2Lip dubs of TEST-split clips (speaker-disjoint from everything trained on),
then forward each fake through the WhatsApp chain (Gen 1-5) like a real fake would be. Never used for training/val/cal.
  full     new speech (from another test clip) put in the whole mouth
  partial  a 2-3 s window dubbed, the rest original
Output: data/attacks/wav2lip/<clip>__<kind>/gen_0.mp4 (the fake) + gen_1..5.mp4 + label.json ; resumable.
Usage: python -m ml.redteam.wav2lip_batch [--limit 96]"""
import argparse, hashlib, json, shutil
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path("data/attacks/wav2lip")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=96); a = ap.parse_args()
    from ml.redteam.wav2lip_dub import dub_clip
    from ml.forwardchain.emulate import emulate
    from satya_core.preprocess import decode
    man = pd.read_parquet("data/manifests/video_real.parquet"); man = man[man.split == "test"]
    rows = [r for r in man.to_dict("records") if (Path("data/cache_v") / r["clip_id"] / "meta.json").exists()][: a.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    wavs = {}
    def wav_of(p):
        if p not in wavs: wavs[p] = decode(p, max_side=64)[1]        # audio only matters; tiny frames keep it cheap
        return wavs[p]
    n = {"ok": 0, "skip": 0}
    for i, r in enumerate(rows):
        cid = r["clip_id"]; rng = np.random.default_rng(int(hashlib.md5(cid.encode()).hexdigest()[:8], 16))
        kind = "full" if i % 5 < 3 else "partial"; d = OUT / f"{cid}__{kind}"
        if (d / "label.json").exists(): n["ok"] += 1; continue
        try:
            src_audio = wav_of(rows[(i + 7) % len(rows)]["path"])
            if len(src_audio) < 3 * 16000: raise ValueError("donor audio too short")
            d.mkdir(parents=True, exist_ok=True)
            win = None if kind == "full" else (float(rng.uniform(0.5, 1.5)), float(rng.uniform(3.0, 4.0)))
            info = dub_clip(r["path"], src_audio, str(d / "gen_0.mp4"), window=win)
            emulate(str(d / "gen_0.mp4"), d, gens=5, seed=int(rng.integers(1 << 30)))   # writes gen_1..5.mp4 (+ params.json); gen_0 stays the fake itself
            (d / "label.json").write_text(json.dumps({"clip": cid, "kind": kind, "window": win, "generator": "wav2lip_gan", "speaker": r["speaker"], "lang": r["lang"]}))
            n["ok"] += 1
        except Exception as e:
            shutil.rmtree(d, ignore_errors=True); n["skip"] += 1; print("skip", cid[:24], repr(e)[:90], flush=True)
        if (i + 1) % 10 == 0: print(f"{i+1}/{len(rows)} {n}", flush=True)
    print("WAV2LIP_DONE", n)

if __name__ == "__main__":
    main()
