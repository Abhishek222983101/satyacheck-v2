"""P7.4 — plots for the benchmark wall. Reads measured JSONs + per-sample scores from
the manifests (re-fusing via the saved Nyaya checkpoint), writes PNGs + the same series
as JSON for the web app to render.  python -m ml.eval.plots
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

R = Path("ml/reports")
C = {"satya": "#22282B", "sync": "#12A6C4", "splice": "#0e9f6e", "face": "#D9B21B", "sn": "#C23A2B", "pixel": "#7C3AED"}


def _roc(y, s, label, colour, ls="-"):
    fpr, tpr, _ = roc_curve(y, s)
    a = roc_auc_score(y, s)
    plt.plot(fpr, tpr, color=colour, ls=ls, lw=1.8, label=f"{label} ({a:.3f})")
    return a


def main():
    series: dict = {"roc": {}, "auc_by_gen": {}, "latency_ms": {}}
    ny = json.loads((R / "nyaya.json").read_text())
    ho = json.loads((R / "heldout_wav2lip.json").read_text()) if (R / "heldout_wav2lip.json").exists() else None
    ab = json.loads((R / "ablations.json").read_text()) if (R / "ablations.json").exists() else None

    # ---- in-domain: re-fuse test rows from branch scores + saved checkpoint
    from ml.train.nyaya import apply
    import torch
    ck = torch.load("models/nyaya.pt", map_location="cpu", weights_only=True)
    from ml.train.nyaya import Nyaya
    net = Nyaya(ck.get("mode", "mean")); net.load_state_dict(ck["model"]); net.eval()
    T = float(ck["T"])
    test = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/manifests/nyaya_test_*of*.parquet"))], ignore_index=True)
    p, _, y = apply(net, test, T)

    plt.figure(figsize=(6.4, 5.2))
    a_fused = _roc(y, p, "SatyaCheck (fused)", C["satya"])
    for col, lab, colo in (("m1_score", "SyncFormer alone", C["sync"]), ("m2_splice", "SpliceLoc alone", C["splice"]), ("m4_score", "FaceTrace alone", C["face"])):
        m = test[col].notna()
        if m.sum() > 50 and len(set(y[m])) > 1:
            _roc(y[m], test.loc[m, col], lab, colo, "--")
    if ab:
        po = ab["baselines_in_domain"].get("mean-of-branches")
        if po:
            plt.plot([], [], label=f"mean-of-branches ({po:.3f})", color="grey")
    plt.plot([0, 1], [0, 1], ls=":", c="#999", lw=1)
    plt.xlabel("false positive rate"); plt.ylabel("true positive rate")
    plt.title(f"In-domain ROC (nyaya test, n={len(y)})"); plt.legend(fontsize=8); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(R / "roc_in_domain.png", dpi=150); plt.close()
    series["roc"]["in_domain"] = {"auc_fused": round(float(a_fused), 4), "n": int(len(y))}

    # AUC vs generation (in-domain fused + each ablation-free branch)
    for gg in sorted(test.gen.unique()):
        m = (test.gen == gg)
        if m.sum() > 30 and len(set(y[m])) > 1:
            series["auc_by_gen"].setdefault("in_domain", {})[str(int(gg))] = round(float(roc_auc_score(y[m], p[m])), 4)
    # held-out generator
    if ho:
        for gg, d in ho["by_gen"].items():
            if d.get("auc_nyaya"):
                series["auc_by_gen"].setdefault("unseen_wav2lip", {})[str(gg)] = round(d["auc_nyaya"], 4)

    if ab:
        plt.figure(figsize=(7, 4))
        labs, vals = [], []
        for k, v in ab["in_domain_ablations"].items():
            labs.append(k.replace("drop_", "−").replace("(sync)", " M1").replace("(splice+synth)", " M2").replace("(face)", " M4").replace("only_m1", "M1 only"))
            vals.append(v["auc"])
        plt.barh(labs, vals, color=[C["satya"] if i == 0 else "#8aa" for i in range(len(vals))])
        for i, v in enumerate(vals):
            plt.text(v + .01, i, f"{v:.3f}", va="center", fontsize=9)
        plt.xlabel("TEST AUC"); plt.title("Ablations (refit without each branch)"); plt.xlim(0.5, 1.0); plt.grid(alpha=.3, axis="x")
        plt.tight_layout(); plt.savefig(R / "ablations.png", dpi=150); plt.close()
        series["roc"]["ablations"] = {k: round(v["auc"], 4) for k, v in ab["in_domain_ablations"].items()}

    # ---- held-out per-sample ROC
    if ho:
        df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/manifests/heldout_*of*.parquet"))], ignore_index=True)
        plt.figure(figsize=(6.4, 5.2))
        _roc(df.label, df.p_fake, "SatyaCheck (fused)", C["satya"])
        for col, lab, colo in (("m1", "SyncFormer", C["sync"]), ("m2_splice", "SpliceLoc", C["splice"]), ("m4", "FaceTrace", C["face"]), ("sn", "SyncNet zero-shot", C["sn"])):
            m = df[col].notna()
            if m.sum() > 50:
                _roc(df.label[m], df.loc[m, col], lab, colo, "--")
        plt.plot([0, 1], [0, 1], ls=":", c="#999", lw=1)
        plt.xlabel("false positive rate"); plt.ylabel("true positive rate")
        plt.title(f"Unseen generator: Wav2Lip dubs (n={len(df)})"); plt.legend(fontsize=8); plt.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(R / "roc_unseen.png", dpi=150); plt.close()
        series["roc"]["unseen"] = {"n": int(len(df)), "fake_called_REAL_by_gen": {str(k): v.get("fake_called_REAL") for k, v in ho["by_gen"].items()}}
        plt.figure(figsize=(6.4, 3.4))
        plt.hist(df.ms, bins=40, color=C["satya"], alpha=.85)
        plt.axvline(df.ms.quantile(.5), color=C["sync"], ls="--", label=f"p50 {df.ms.median():.0f} ms")
        plt.axvline(df.ms.quantile(.95), color=C["sn"], ls="--", label=f"p95 {df.ms.quantile(.95):.0f} ms")
        plt.xlabel("per-clip latency (ms)"); plt.legend(); plt.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(R / "latency.png", dpi=150); plt.close()
        series["latency_ms"] = {"p50": round(float(df.ms.median())), "p95": round(float(df.ms.quantile(.95)))}

    (R / "plots_series.json").write_text(json.dumps(series, indent=2, default=str))
    print("plots ->", R, "| AUC-vs-gen:", json.dumps(series["auc_by_gen"]))


if __name__ == "__main__":
    main()
