"""P3.4 — full held-out evaluation of SpliceLoc on the TEST split (identity/sentence-disjoint from train/val/cal).

Sets (all deterministic: sample i is a pure function of (seed, i)):
  mixed[g]   g=0..5 forwarded generations, natural mix of real / synthetic / spliced clips
  per-op[g]  real clips only, each attack type forced (delete / swap / duplicate / insert-TTS / insert-real) vs clean real
Threshold for boundary peaks comes from validation (stored in the checkpoint) — never tuned on test.
Writes ml/reports/spliceloc.json + PNGs, and logs everything to W&B (group M2-eval).
Usage: python -m ml.eval.eval_spliceloc [--ckpt ml/runs/spliceloc/best.pt] [--n 400]"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from ml.data.dataset_splice import SpliceDataset, LANGS, DSETS, OPS
from ml.train.spliceloc import SpliceLoc, eer, pick_peaks, gt_centres, boundary_f1

REP = Path("ml/reports"); REP.mkdir(parents=True, exist_ok=True)


def _auc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 and len(y) > 10 else None


@torch.no_grad()
def collect(model, ds, dev, workers=6, bs=16):
    model.eval(); c = {k: [] for k in ("fy", "fs", "bp", "bg", "ys", "yp", "us", "up", "gen", "op", "lang", "ds")}
    for b in DataLoader(ds, batch_size=bs, num_workers=workers, shuffle=False):
        wav = b["wav"].to(dev); nvs = (b["valid"].sum(1) * 320).long().to(dev).clamp(max=wav.shape[1])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(wav, nvs, valid=b["valid"].to(dev))
        T = o["boundary"].shape[1]
        ps = torch.softmax(o["frame"].float(), -1)[..., 1].cpu().numpy(); pb = torch.sigmoid(o["boundary"].float()).cpu().numpy()
        pu = torch.sigmoid(o["utt"].float()).cpu().numpy(); v = b["valid"][:, :T].numpy() > 0.5
        for i in range(len(wav)):
            c["fy"].append(b["frame_spoof"][i, :T].numpy()[v[i]]); c["fs"].append(ps[i][v[i]])
            c["bp"].append(pb[i] * v[i]); c["bg"].append(gt_centres(b["boundary"][i, :T].numpy()))
        c["ys"] += b["utt_synth"].tolist(); c["yp"] += b["utt_splice"].tolist(); c["us"] += pu[:, 0].tolist(); c["up"] += pu[:, 1].tolist()
        c["gen"] += b["gen"].tolist(); c["op"] += b["op"].tolist(); c["lang"] += b["lang_id"].tolist(); c["ds"] += b["ds_id"].tolist()
    return c


def metrics(c, thr):
    fy, fs = np.concatenate(c["fy"]), np.concatenate(c["fs"])
    preds = [pick_peaks(p, thr) for p in c["bp"]]
    f80 = boundary_f1(preds, c["bg"], 4); f40 = boundary_f1(preds, c["bg"], 2)
    # boundary metrics only make sense on clips that HAVE a splice
    idx = [i for i, g in enumerate(c["bg"]) if len(g)]
    f80s = boundary_f1([preds[i] for i in idx], [c["bg"][i] for i in idx], 4) if idx else (None,) * 3
    return {"n": len(c["gen"]), "frame_eer": eer(fy, fs), "boundary_f1_80ms": f80[0], "boundary_f1_40ms": f40[0],
            "boundary_f1_80ms_spliced_only": f80s[0], "boundary_prec_80ms": f80[1], "boundary_rec_80ms": f80[2],
            "auc_splice": _auc(c["yp"], c["up"]), "auc_synth": _auc(c["ys"], c["us"])}


def sub(c, mask):
    m = np.nonzero(mask)[0]; out = {}
    for k, v in c.items(): out[k] = [v[i] for i in m]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default="ml/runs/spliceloc/best.pt"); ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--manifest", default="data/manifests/audio_split.parquet"); ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args(); dev = "cuda"
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=True); thr = float(ck["metrics"].get("boundary_thr", 0.5))
    model = SpliceLoc().to(dev); model.load_state_dict(ck["model"]); model.eval()
    R = {"ckpt": a.ckpt, "epoch": ck["epoch"], "boundary_thr_from_val": thr, "n_per_set": a.n, "mixed_by_gen": {}, "per_op": {}, "by_lang": {}, "by_dataset": {}}
    t0 = time.time(); allc = None
    for g in range(6):
        c = collect(model, SpliceDataset(a.manifest, "test", a.n, True, seed=777 + g, splice_p=0.5, force_gen=g), dev)
        R["mixed_by_gen"][g] = metrics(c, thr)
        allc = c if allc is None else {k: allc[k] + c[k] for k in c}
        print(f"gen {g}:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in R["mixed_by_gen"][g].items()}, flush=True)
    R["mixed_all"] = metrics(allc, thr)
    lang, dsid = np.array(allc["lang"]), np.array(allc["ds"])
    for i, l in enumerate(LANGS):
        if (lang == i).sum() > 20: R["by_lang"][l] = metrics(sub(allc, lang == i), thr)
    for i, d in enumerate(DSETS):
        if (dsid == i).sum() > 20: R["by_dataset"][d] = metrics(sub(allc, dsid == i), thr)
    # synthetic detection WITHIN a single corpus (no corpus shortcut): indictts real-vs-tts only
    m = dsid == DSETS.index("indictts"); R["synth_within_corpus_indictts_auc"] = _auc(np.array(allc["ys"])[m], np.array(allc["us"])[m])
    # per attack type: positives = that op on real clips, negatives = clean real clips (same gen)
    for g in (0, 3, 5):
        neg = collect(model, SpliceDataset(a.manifest, "test", a.n, True, seed=900 + g, splice_p=0.0, force_gen=g), dev)
        neg = sub(neg, np.array(neg["ys"]) == 0)
        for op in OPS:
            pos = collect(model, SpliceDataset(a.manifest, "test", a.n, True, seed=1000 + g, splice_p=1.0, force_gen=g, force_op=op), dev)
            pos = sub(pos, np.array(pos["yp"]) == 1)
            both = {k: pos[k] + neg[k] for k in pos}
            R["per_op"].setdefault(op, {})[g] = {"n_pos": len(pos["gen"]), **{k: v for k, v in metrics(both, thr).items() if k in ("auc_splice", "boundary_f1_80ms_spliced_only")}}
        print("per-op gen", g, {op: R["per_op"][op][g] for op in OPS}, flush=True)
    R["eval_seconds"] = time.time() - t0
    (REP / "spliceloc.json").write_text(json.dumps(R, indent=2, default=float)); print("wrote", REP / "spliceloc.json")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    gs = list(range(6)); fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for key, lab, col in (("auc_splice", "splice detection AUC", "#C2338F"), ("auc_synth", "synthetic-voice AUC", "#22282B")):
        ax[0].plot(gs, [R["mixed_by_gen"][g][key] for g in gs], marker="o", color=col, label=lab)
    ax[0].set_ylim(0.4, 1.01); ax[0].set_xlabel("WhatsApp forwards"); ax[0].set_title("AUC vs generation"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(gs, [R["mixed_by_gen"][g]["boundary_f1_80ms_spliced_only"] for g in gs], marker="o", color="#C2338F", label="±80 ms")
    ax[1].plot(gs, [R["mixed_by_gen"][g]["boundary_f1_40ms"] for g in gs], marker="s", color="#12A6C4", label="±40 ms (all clips)")
    ax[1].set_ylim(0, 1); ax[1].set_xlabel("WhatsApp forwards"); ax[1].set_title("Cut-point localisation F1"); ax[1].legend(); ax[1].grid(alpha=.3)
    w = 0.25
    for j, g in enumerate((0, 3, 5)):
        ax[2].bar(np.arange(len(OPS)) + (j - 1) * w, [R["per_op"][o][g]["boundary_f1_80ms_spliced_only"] or 0 for o in OPS], w, label=f"Gen-{g}")
    ax[2].set_xticks(range(len(OPS))); ax[2].set_xticklabels([o.replace("insert_", "ins-") for o in OPS], fontsize=8); ax[2].set_ylim(0, 1)
    ax[2].set_title("Localisation F1 by attack"); ax[2].legend(); ax[2].grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.savefig(REP / "spliceloc_summary.png", dpi=150)
    if not a.no_wandb:
        import wandb
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "satyacheck"), group="M2-eval", name=f"spliceloc-test-ep{ck['epoch']}", config={"ckpt": a.ckpt, "thr": thr})
        run.log({"eval/summary": wandb.Image(str(REP / "spliceloc_summary.png"))})
        for g in gs: run.log({f"eval/auc_splice": R["mixed_by_gen"][g]["auc_splice"], f"eval/auc_synth": R["mixed_by_gen"][g]["auc_synth"], "eval/boundary_f1_40ms": R["mixed_by_gen"][g]["boundary_f1_40ms"], "generation": g})
        run.summary.update({"test/auc_splice_all": R["mixed_all"]["auc_splice"], "test/auc_synth_all": R["mixed_all"]["auc_synth"], "test/frame_eer_all": R["mixed_all"]["frame_eer"],
                            "test/synth_auc_within_indictts": R["synth_within_corpus_indictts_auc"]})
        run.finish()

if __name__ == "__main__":
    main()
