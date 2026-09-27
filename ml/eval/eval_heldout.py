"""P7 (core) — held-out generator evaluation, END TO END on real video FILES.

Negatives: the untouched test clips at Gen 0..5 (originals + WhatsApp-style ladder files).
Positives: Wav2Lip dubs of the same test clips (a lip-sync generator none of our models ever saw), forwarded through the same
           chain. `full` = whole mouth re-dubbed with other speech, `partial` = a 2-3 s window.
Every file goes through satya_core.pipeline.analyze_path (preprocess -> M1/M2/M3/M4 -> Nyaya verdict) — the exact runtime path —
and, for comparison, the classic SyncNet zero-shot detector (conf-based) on the same face crops and audio.
Gen-0 caveat: fake Gen 0 is our single re-encode of the dub, real Gen 0 is the original upload; Gens >= 1 share the same chain.

Usage: python -m ml.eval.eval_heldout --part 0/3 --gens 0,1,3,5   (x3 in parallel)  then  python -m ml.eval.eval_heldout --report"""
from __future__ import annotations

import argparse, glob, json, os, time
from pathlib import Path

import numpy as np
import pandas as pd

ATT, GENS, MAN = Path("data/attacks/wav2lip"), Path("data/gens_video"), Path("data/manifests")


def jobs(gens):
    man = pd.read_parquet(MAN / "video_real.parquet").set_index("clip_id"); out = []
    for d in sorted(ATT.glob("*__*")):
        if not (d / "label.json").exists(): continue
        lab = json.loads((d / "label.json").read_text()); cid = lab["clip"]
        for g in gens:
            fake = d / f"gen_{g}.mp4"
            real = man.loc[cid, "path"] if g == 0 else str(GENS / cid / f"gen_{g}.mp4")
            if fake.exists() and Path(real).exists():
                out.append(dict(clip=cid, gen=g, kind=lab["kind"], path=str(fake), label=1, lang=lab["lang"], speaker=lab["speaker"]))
                out.append(dict(clip=cid, gen=g, kind="real", path=str(real), label=0, lang=lab["lang"], speaker=lab["speaker"]))
    seen, uniq = set(), []
    for j in out:                                                     # a real file is shared by fakes of the same clip: score once per (clip, gen)
        k = (j["path"], j["label"])
        if k not in seen: seen.add(k); uniq.append(j)
    return uniq


def syncnet_score(T, pre_path, pre):
    """Classic zero-shot: SyncNet offset/confidence on the same clip. Score high = out of sync (fake)."""
    from ml.teachers.syncnet.teacher import syncnet_crops
    from satya_core.preprocess import decode
    if pre.pts_norm is None or pre.meta.get("no_face") or not pre.meta.get("has_audio"): return None
    fr, wav, _ = decode(pre_path); H, W = fr.shape[1:3]; n = min(len(fr), len(pre.pts_norm))
    vf, af = T.embed(syncnet_crops(fr[:n], pre.pts_norm[:n].astype(np.float32) * [W, H]), wav)
    if len(vf) < 20: return None
    _, off, conf = T.profile(vf, af)
    return float(-conf if abs(off) <= 2 else conf)


def run(args):
    from satya_core.pipeline import Models, analyze_path
    from satya_core.verdict import Fuser
    models = Models("cuda"); fuser = Fuser("models/nyaya.pt") if Path("models/nyaya.pt").exists() else None
    from ml.teachers.syncnet.teacher import SyncNetTeacher
    T = SyncNetTeacher("cuda")
    pi, pn = map(int, args.part.split("/")); J = jobs([int(g) for g in args.gens.split(",")])[pi::pn]; rows = []; t0 = time.time()
    for i, j in enumerate(J):
        try:
            import satya_core.pipeline as P
            from satya_core.preprocess import preprocess
            from satya_core.features_codec import extract
            pre = preprocess(j["path"]); vec, names = extract(j["path"]); b = P.analyze_arrays(models, pre, vec, names)
            v = fuser(b, pre.meta) if fuser else {}
            r = dict(j, m1=b.m1.get("clip_score"), m1_off=b.m1.get("offset_ms"), m2_splice=b.m2.get("p_spliced"), m2_synth=b.m2.get("p_synthetic"), m4=b.m4.get("clip_score"),
                     gen_hat=b.m3["gen_hat"], p_fake=v.get("p_fake"), verdict=v.get("label"), reason=v.get("reason"), sn=syncnet_score(T, j["path"], pre),
                     m1_ok=b.m1["status"] == "ok", face=pre.meta["face_presence"], speech=pre.meta["speech_ratio"], ms=sum(b.timings_ms.values()))
            rows.append(r)
        except Exception as e:
            print("skip", Path(j["path"]).parent.name[:24], j["gen"], repr(e)[:90], flush=True)
        if (i + 1) % 20 == 0: print(f"{i+1}/{len(J)} {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_parquet(MAN / f"heldout_{pi}of{pn}.parquet", index=False); print("wrote part", len(rows))


def report():
    from sklearn.metrics import roc_auc_score
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(MAN / "heldout_*of*.parquet")))], ignore_index=True)
    R = {"n_files": len(df), "n_fake": int(df.label.sum()), "n_real": int((1 - df.label).sum()), "by_gen": {}, "by_kind": {}}
    def auc(d, col):
        d = d.dropna(subset=[col]); return float(roc_auc_score(d.label, d[col])) if len(set(d.label)) > 1 and len(d) > 10 else None
    for g, d in df.groupby("gen"):
        vd = d[d.verdict.notna()] if "verdict" in d else d
        fk, rl = vd[vd.label == 1], vd[vd.label == 0]
        R["by_gen"][int(g)] = {"n": len(d), "auc_nyaya": auc(d, "p_fake"), "auc_syncformer": auc(d, "m1"), "auc_facetrace": auc(d, "m4"), "auc_spliceloc": auc(d, "m2_splice"),
                                "auc_syncnet_zero_shot": auc(d, "sn"),
                                "detected_as_FAKE": float((fk.verdict == "FAKE").mean()) if len(fk) else None, "fake_called_REAL": float((fk.verdict == "REAL").mean()) if len(fk) else None,
                                "fake_abstained": float((fk.verdict == "SUSPECT").mean()) if len(fk) else None,
                                "real_called_FAKE": float((rl.verdict == "FAKE").mean()) if len(rl) else None, "real_abstained": float((rl.verdict == "SUSPECT").mean()) if len(rl) else None}
    for k in ("full", "partial"):
        d = df[df.kind.isin([k, "real"])]; R["by_kind"][k] = {"auc_nyaya": auc(d, "p_fake"), "auc_syncformer": auc(d, "m1"), "auc_syncnet_zero_shot": auc(d, "sn"), "auc_facetrace": auc(d, "m4")}
    R["latency_ms"] = {"mean": float(df.ms.mean()), "p95": float(df.ms.quantile(0.95))}
    Path("ml/reports").mkdir(exist_ok=True); json.dump(R, open("ml/reports/heldout_wav2lip.json", "w"), indent=2, default=float)
    print(json.dumps({g: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items()} for g, x in R["by_gen"].items()}, indent=1)); print("by kind", R["by_kind"])
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    gs = sorted(R["by_gen"]); fig, ax = plt.subplots(figsize=(6.5, 4))
    for key, lab, col, ls in (("auc_nyaya", "SatyaCheck (fused)", "#22282B", "-"), ("auc_syncformer", "SyncFormer alone", "#12A6C4", "-"), ("auc_syncnet_zero_shot", "SyncNet zero-shot (baseline)", "#C23A2B", "--"), ("auc_facetrace", "FaceTrace alone", "#D9B21B", ":")):
        ys = [R["by_gen"][g][key] for g in gs]
        if any(y is not None for y in ys): ax.plot(gs, [np.nan if y is None else y for y in ys], marker="o", label=lab, color=col, ls=ls)
    ax.set_ylim(0.3, 1.02); ax.set_xlabel("WhatsApp forwards"); ax.set_ylabel("AUC vs real (Wav2Lip, unseen generator)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig("ml/reports/heldout_wav2lip.png", dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--part", default="0/1"); ap.add_argument("--gens", default="0,1,3,5"); ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report() if a.report else run(a)
