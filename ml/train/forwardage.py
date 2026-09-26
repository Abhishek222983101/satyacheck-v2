"""M3 ForwardAge — how many WhatsApp-style forwards has this clip been through? (blueprint §4.6)

Features: satya_core.features_codec (75-d codec forensics). Model: ordinal LightGBM — five binary classifiers
P(gen > k), k = 0..4 -> full distribution over Gen 0..5 and expected age  E[g] = sum_k P(g > k).
Ground truth is free (the ForwardChain ladder). Splits are speaker-disjoint (manifest).

HONESTY: train/test clips are forwarded by OUR emulator. Accuracy here measures 'can it recover the emulator's
generation', NOT 'can it age real WhatsApp files'. The real-world number comes only from T-WA (clips truly forwarded on
phones): `--twa data/raw/whatsapp` (dirs of <clip>/<hop>.mp4) fills `twa` in the report. Until then say 'emulated'.

Usage:  python -m ml.train.forwardage --build     # feature table (parallel, resumable)
        python -m ml.train.forwardage --train     # fit + evaluate + write ml/reports/forwardage.json, models/forwardage.txt"""
from __future__ import annotations

import argparse, json, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

TABLE = Path("data/manifests/codec_feats.parquet")
GENS = Path("data/gens_video")


def _one(args):
    cid, gen, path, split, lang = args
    try:
        from satya_core.features_codec import extract
        v, names = extract(path)
        return dict(clip_id=cid, gen=gen, split=split, lang=lang, **dict(zip(names, map(float, v))))
    except Exception as e:
        return dict(clip_id=cid, gen=gen, split=split, lang=lang, error=repr(e)[:80])


def build(workers: int = 4):
    df = pd.read_parquet("data/manifests/video_real.parquet")
    jobs = []
    for r in df.itertuples():
        if not (Path("data/cache_v") / r.clip_id / "meta.json").exists():
            continue
        jobs.append((r.clip_id, 0, r.path, r.split, r.lang))
        for k in range(1, 6):
            p = GENS / r.clip_id / f"gen_{k}.mp4"
            if p.exists():
                jobs.append((r.clip_id, k, str(p), r.split, r.lang))
    done = pd.read_parquet(TABLE) if TABLE.exists() else pd.DataFrame(columns=["clip_id", "gen"])
    have = set(zip(done.clip_id, done.gen)); jobs = [j for j in jobs if (j[0], j[1]) not in have]
    print(f"{len(jobs)} feature extractions to do ({len(have)} cached)", flush=True)
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(_one, jobs, chunksize=8), 1):
            if "error" not in r: rows.append(r)
            if i % 200 == 0: print(i, "/", len(jobs), flush=True)
    out = pd.concat([done, pd.DataFrame(rows)], ignore_index=True) if rows else done
    TABLE.parent.mkdir(exist_ok=True); out.to_parquet(TABLE, index=False); print("table:", len(out), "rows")


def distribution(P):
    """P (n,5): P(g>k) -> (n,6) probabilities over Gen 0..5, made monotone (cummin) so they are valid."""
    P = np.minimum.accumulate(np.clip(P, 0, 1), axis=1)
    edges = np.concatenate([np.ones((len(P), 1)), P, np.zeros((len(P), 1))], 1)
    return edges[:, :-1] - edges[:, 1:]


def train():
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    df = pd.read_parquet(TABLE)
    feat = [c for c in df.columns if c not in ("clip_id", "gen", "split", "lang", "error")]
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    print(f"rows train {len(tr)} val {len(va)} test {len(te)} | features {len(feat)}")
    models = []
    for k in range(5):
        m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.04, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.7, reg_lambda=2.0, verbose=-1, n_jobs=4)
        m.fit(tr[feat], (tr.gen > k).astype(int), eval_set=[(va[feat], (va.gen > k).astype(int))], callbacks=[lgb.early_stopping(40, verbose=False)])
        models.append(m)
    def predict(d):
        P = np.stack([m.predict_proba(d[feat])[:, 1] for m in models], 1)
        D = distribution(P); return D, (D * np.arange(6)).sum(1)
    R = {"n_features": len(feat), "note": "EMULATED forwards (ForwardChain). Not a measurement of real WhatsApp files.", "splits": {}}
    for name, d in (("val", va), ("test", te)):
        D, E = predict(d); g = d.gen.values; hard = D.argmax(1)
        R["splits"][name] = {"n": len(d), "mae_generations": float(np.abs(E - g).mean()), "acc_exact": float((hard == g).mean()),
                             "acc_within_1": float((np.abs(hard - g) <= 1).mean()),
                             "auc_forwarded_vs_original": float(roc_auc_score(g > 0, E)) if len(set(g > 0)) > 1 else None,
                             "mae_by_gen": {int(k): float(np.abs(E[g == k] - k).mean()) for k in range(6) if (g == k).any()},
                             "confusion": pd.crosstab(pd.Series(g, name="true"), pd.Series(hard, name="pred")).to_dict()}
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in R["splits"][name].items() if k not in ("confusion", "mae_by_gen")}, "| MAE by gen", {k: round(v, 2) for k, v in R["splits"][name]["mae_by_gen"].items()})
    # leave-one-language-out style sanity: does it hold on languages with few clips?
    D, E = predict(te)
    R["test_mae_by_lang"] = {l: float(np.abs(E[(te.lang == l).values] - te.gen.values[(te.lang == l).values]).mean()) for l in te.lang.unique() if (te.lang == l).sum() > 30}
    # feature importance (gain) averaged over the five binary models
    imp = np.mean([m.booster_.feature_importance("gain") for m in models], 0); imp = imp / imp.sum()
    R["top_features"] = [(feat[i], float(imp[i])) for i in np.argsort(-imp)[:12]]; print("top features:", [(a, round(b, 3)) for a, b in R["top_features"][:6]])
    Path("models").mkdir(exist_ok=True); Path("ml/reports").mkdir(parents=True, exist_ok=True)
    for k, m in enumerate(models): m.booster_.save_model(f"models/forwardage_gt{k}.txt")
    json.dump({"features": feat}, open("models/forwardage_features.json", "w"))
    json.dump(R, open("ml/reports/forwardage.json", "w"), indent=2, default=float)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8)); D, E = predict(te)
    ax[0].boxplot([E[te.gen.values == k] for k in range(6)], positions=range(6)); ax[0].plot([-.5, 5.5], [-.5, 5.5], "k--", lw=1)
    ax[0].set_xlabel("true forwards (emulated)"); ax[0].set_ylabel("predicted"); ax[0].set_title("ForwardAge on held-out speakers"); ax[0].grid(alpha=.3)
    ax[1].barh([a for a, _ in R["top_features"][:10]][::-1], [b for _, b in R["top_features"][:10]][::-1], color="#22282B"); ax[1].set_title("What it listens to (gain)")
    plt.tight_layout(); plt.savefig("ml/reports/forwardage.png", dpi=150)
    if os.environ.get("WANDB_API_KEY"):
        import wandb
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "satyacheck"), group="M3-ForwardAge", name="forwardage-lgbm", config={"features": len(feat), "emulated": True})
        run.log({"test/mae_generations": R["splits"]["test"]["mae_generations"], "test/acc_within_1": R["splits"]["test"]["acc_within_1"], "plot": wandb.Image("ml/reports/forwardage.png")}); run.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--build", action="store_true"); ap.add_argument("--train", action="store_true"); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.build: build(a.workers)
    if a.train: train()
