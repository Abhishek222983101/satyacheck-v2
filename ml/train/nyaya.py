"""M5 Nyaya — compression-gated fusion, temperature scaling, Mondrian split-conformal abstention (blueprint §4.8).

  branch scores (M1 sync, M2 splice, M2 synthetic-voice, M4 face) --logit--> per-branch affine (w_i, b_i)
  context (gen_hat, face_presence, speech_ratio)  -> softmax gate over the AVAILABLE branches
  logit_fake = sum_i g_i (w_i s_i + b_i) + residual(z)            trained on VAL with random branch dropout
  temperature T fitted on CAL;  conformal q̂ per generation bucket fitted on CAL (alpha = 0.05)
  prediction set C(x) = { y : 1 - p_y(x) <= q̂_bucket } -> {FAKE}=FAKE  {REAL}=REAL  else SUSPECT (with a reason code)
Guarantee (marginal, per bucket, exchangeable data): P(true label in C(x)) >= 1 - alpha.  It is a statement about the
committed answers' reliability, not about accuracy on unseen attack families — see the claim guard (blueprint §4.11).

Usage: python -m ml.train.nyaya      (needs data/manifests/nyaya_{val,cal,test}_*.parquet from ml.train.nyaya_data)"""
from __future__ import annotations

import glob, json, math, os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

BR = ["m1_score", "m2_splice", "m2_synth", "m4_score"]
OKS = {"m1_score": "m1_ok", "m2_splice": "m2_ok", "m2_synth": "m2_ok", "m4_score": "m4_ok"}
EPS = 1e-4


def logit(p):
    p = np.clip(np.asarray(p, np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def bucket_of(gen_hat):
    """generation bucket used for Mondrian calibration: 0 = Gen 0-1, 1 = Gen 2-3, 2 = Gen 4-5"""
    g = np.asarray(gen_hat, np.float64)
    return np.where(g < 1.5, 0, np.where(g < 3.5, 1, 2))


def load(split):
    fs = sorted(glob.glob(f"{os.environ.get('SATYA_MANIFEST_DIR', 'data/manifests')}/nyaya_{split}_*of*.parquet"))
    if not fs:
        raise FileNotFoundError(f"no nyaya_{split}_*.parquet — run ml.train.nyaya_data --split {split}")
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def tensors(df):
    S = np.stack([np.where(df[OKS[b]].values == 1, logit(df[b].fillna(0.5).values), 0.0) for b in BR], 1)
    M = np.stack([df[OKS[b]].values.astype(np.float64) for b in BR], 1)
    C = np.stack([df["gen_hat"].values / 5.0, df["face_presence"].values, df["speech_ratio"].values.clip(0, 1)], 1)
    y = df["label"].values.astype(np.float64)
    return [torch.tensor(a, dtype=torch.float32) for a in (S, M, C, y)]


class Nyaya(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(4)); self.b = nn.Parameter(torch.zeros(4))
        self.gate = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 4))
        self.res = nn.Sequential(nn.Linear(7, 16), nn.GELU(), nn.Linear(16, 1))
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.res[-1].weight); nn.init.zeros_(self.res[-1].bias)       # start as a pure gated mean

    def forward(self, S, M, C):
        g = self.gate(C).masked_fill(M < 0.5, -1e4)
        g = torch.softmax(g, -1) * (M.sum(-1, keepdim=True) > 0)                     # nothing available -> zero evidence
        z = (g * (self.w * S + self.b)).sum(-1)
        return z + self.res(torch.cat([C, M], -1)).squeeze(-1) * 0.5 + self.bias, g


def fit(dfv, epochs=600, seed=0, p_drop=0.25):
    torch.manual_seed(seed); np.random.seed(seed)
    spk = dfv.speaker.astype(str).values; u = np.unique(spk); rng = np.random.default_rng(seed); rng.shuffle(u)
    hold = np.isin(spk, u[: max(1, len(u) // 5)])                                   # speaker-disjoint early-stopping split
    S, M, C, y = tensors(dfv); Str, Mtr, Ctr, ytr = S[~hold], M[~hold], C[~hold], y[~hold]; Sh, Mh, Ch, yh = S[hold], M[hold], C[hold], y[hold]
    net = Nyaya(); opt = torch.optim.AdamW(net.parameters(), lr=3e-2, weight_decay=1e-3); best, best_sd, bad = 1e9, None, 0
    for ep in range(epochs):
        net.train()
        keep = (torch.rand_like(Mtr) > p_drop).float() * Mtr                          # random branch dropout (mute / faceless clips)
        keep[keep.sum(1) == 0] = Mtr[keep.sum(1) == 0]
        z, _ = net(Str * keep, keep, Ctr)
        loss = F.binary_cross_entropy_with_logits(z, ytr) + 1e-3 * sum((p ** 2).sum() for p in net.res.parameters())
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 0:
            net.eval()
            with torch.no_grad():
                vl = F.binary_cross_entropy_with_logits(net(Sh, Mh, Ch)[0], yh).item()
            if vl < best - 1e-4: best, best_sd, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else: bad += 1
            if bad >= 12: break
    net.load_state_dict(best_sd); net.eval(); return net


def apply(net, df, T=1.0, mask_override=None):
    S, M, C, y = tensors(df)
    if mask_override is not None: M = M * torch.tensor(mask_override, dtype=torch.float32); S = S * M
    with torch.no_grad(): z, g = net(S, M, C)
    return torch.sigmoid(z / T).numpy(), g.numpy(), y.numpy()


def nll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6); return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def fit_temperature(net, dfc):
    S, M, C, y = tensors(dfc)
    with torch.no_grad(): z, _ = net(S, M, C)
    best = min(((nll(torch.sigmoid(z / T).numpy(), y.numpy()), T) for T in np.exp(np.linspace(np.log(0.3), np.log(6), 120))))
    return best[1]


def ece(p, y, bins=10):
    e, n = 0.0, len(p)
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (p >= lo) & (p < lo + 1 / bins + (1e-9 if lo + 1 / bins >= 1 else 0))
        if m.any(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def conformal(p_cal, y_cal, gen_cal, alpha):
    """Mondrian split-conformal: per bucket q̂ = ceil((n+1)(1-alpha))/n quantile of the true-label non-conformity 1 - p_y."""
    q = {}; b = bucket_of(gen_cal)
    for k in range(3):
        m = b == k; n = int(m.sum())
        if n < 20: q[k] = None; continue
        s = np.where(y_cal[m] == 1, 1 - p_cal[m], p_cal[m])
        lvl = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n); q[k] = float(np.quantile(s, lvl, method="higher"))
    return q


def predict_sets(p, gen, q):
    b = bucket_of(gen); out = []
    for pi, bi in zip(p, b):
        qk = q.get(int(bi)) if q.get(int(bi)) is not None else 0.5
        s = set()
        if 1 - pi <= qk: s.add("FAKE")
        if pi <= qk: s.add("REAL")
        out.append(s)
    return out


def verdicts(sets):
    return ["FAKE" if s == {"FAKE"} else "REAL" if s == {"REAL"} else "SUSPECT" for s in sets]


def evaluate(net, T, q, alpha, dft, name="test"):
    R = {}
    p, g, y = apply(net, dft, T); gen = dft.gen_hat.values
    sets = predict_sets(p, gen, q); v = np.array(verdicts(sets)); yt = np.where(y == 1, "FAKE", "REAL")
    cov = np.array([yt[i] in sets[i] for i in range(len(y))]); commit = v != "SUSPECT"
    R.update(n=int(len(y)), auc=float(roc_auc_score(y, p)), ece=ece(p, y), nll=nll(p, y), coverage=float(cov.mean()), abstain_rate=float((~commit).mean()),
             accuracy_on_committed=float((v[commit] == yt[commit]).mean()) if commit.any() else None,
             fake_committed_real_rate=float(((v == "REAL") & (y == 1)).sum() / max(1, (y == 1).sum())),
             real_committed_fake_rate=float(((v == "FAKE") & (y == 0)).sum() / max(1, (y == 0).sum())))
    b = bucket_of(gen)
    R["by_bucket"] = {int(k): {"n": int((b == k).sum()), "coverage": float(cov[b == k].mean()), "abstain": float((~commit[b == k]).mean()),
                                "auc": float(roc_auc_score(y[b == k], p[b == k])) if len(set(y[b == k])) > 1 else None} for k in range(3) if (b == k).any()}
    R["by_cond"] = {c: {"n": int((dft.cond.values == c).sum()), "detect_rate": float((v[dft.cond.values == c] == "FAKE").mean()) if c != "real" else None,
                        "abstain": float((v[dft.cond.values == c] == "SUSPECT").mean()), "mean_p_fake": float(p[dft.cond.values == c].mean())} for c in dft.cond.unique()}
    R["by_gen"] = {int(k): {"auc": float(roc_auc_score(y[dft.gen.values == k], p[dft.gen.values == k]))} for k in sorted(dft.gen.unique()) if len(set(y[dft.gen.values == k])) > 1}
    return R, p, v


def main():
    dfv, dfc, dft = load("val"), load("cal"), load("test"); alpha = 0.05
    print(f"val {len(dfv)} cal {len(dfc)} test {len(dft)} samples")
    net = fit(dfv); T = fit_temperature(net, dfc)
    pc, _, yc = apply(net, dfc, T); q = conformal(pc, yc, dfc.gen_hat.values, alpha)
    print("temperature", round(T, 3), "| q̂ per bucket", {k: (round(v, 3) if v else None) for k, v in q.items()})
    R = {"alpha": alpha, "temperature": T, "qhat": q, "n": {"val": len(dfv), "cal": len(dfc), "test": len(dft)}}
    R["test"], p, v = evaluate(net, T, q, alpha, dft)
    print("TEST", {k: (round(x, 4) if isinstance(x, float) else x) for k, x in R["test"].items() if k not in ("by_bucket", "by_cond", "by_gen")})
    print("by bucket", R["test"]["by_bucket"]); print("by cond", {k: {a: (round(b, 2) if isinstance(b, float) else b) for a, b in d.items()} for k, d in R["test"]["by_cond"].items()})
    # ablations: mean of branch logits / best single branch / no gate (uniform) — all evaluated on the same test set
    S, M, C, y = tensors(dft); yv = y.numpy()
    mean_score = (S * M).sum(1) / M.sum(1).clamp(min=1)
    R["baselines"] = {"mean_of_branches_auc": float(roc_auc_score(yv, mean_score.numpy())),
                      **{f"{b}_auc": float(roc_auc_score(yv, S[:, i].numpy())) for i, b in enumerate(BR)}}
    print("baselines", {k: round(v, 3) for k, v in R["baselines"].items()})
    # robustness to a missing branch (mute clip: no M1/M2; faceless: no M1/M4) — same fitted model, masked at inference
    R["missing_branch"] = {}
    for name, mo in (("mute (no sync/splice/synth)", [0, 0, 0, 1]), ("faceless (no sync/face)", [0, 1, 1, 0]), ("no splice/synth", [1, 0, 0, 1])):
        pm, _, ym = apply(net, dft, T, mask_override=[mo] * len(dft))
        # samples whose only informative branch was masked have no evidence -> they stay ~0.5; report AUC on samples with any evidence left
        has = (M.numpy() * np.array(mo)).sum(1) > 0
        R["missing_branch"][name] = {"auc_with_evidence": float(roc_auc_score(ym[has], pm[has])) if has.any() and len(set(ym[has])) > 1 else None, "n": int(has.sum())}
    print("missing branch", R["missing_branch"])
    Path("models").mkdir(exist_ok=True); Path("ml/reports").mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "T": T, "qhat": q, "alpha": alpha, "branches": BR}, "models/nyaya.pt")
    json.dump(R, open("ml/reports/nyaya.json", "w"), indent=2, default=float)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8)); yt = dft.label.values
    ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    for nm, pp, col in (("Nyaya (calibrated)", p, "#22282B"), ("uncalibrated", apply(net, dft, 1.0)[0], "#C2338F")):
        bins = np.linspace(0, 1, 11); idx = np.digitize(pp, bins) - 1; xs = [pp[idx == k].mean() for k in range(10) if (idx == k).any()]; ys = [yt[idx == k].mean() for k in range(10) if (idx == k).any()]
        ax[0].plot(xs, ys, marker="o", label=nm, color=col)
    ax[0].set_title("Reliability (test)"); ax[0].set_xlabel("predicted P(fake)"); ax[0].set_ylabel("observed"); ax[0].legend(); ax[0].grid(alpha=.3)
    bk = R["test"]["by_bucket"]; xs = sorted(bk); ax[1].bar(np.arange(len(xs)) - .2, [bk[k]["coverage"] for k in xs], .4, label="coverage", color="#12A6C4"); ax[1].bar(np.arange(len(xs)) + .2, [bk[k]["abstain"] for k in xs], .4, label="abstain rate", color="#D9B21B")
    ax[1].axhline(1 - alpha, color="k", ls="--", lw=1); ax[1].set_xticks(range(len(xs))); ax[1].set_xticklabels(["Gen 0-1", "Gen 2-3", "Gen 4-5"][: len(xs)]); ax[1].set_ylim(0, 1.05); ax[1].legend(); ax[1].set_title(f"Conformal coverage (target {1-alpha:.0%})")
    cs = list(R["test"]["by_cond"]); ax[2].bar(range(len(cs)), [R["test"]["by_cond"][c]["mean_p_fake"] for c in cs], color="#22282B"); ax[2].set_xticks(range(len(cs))); ax[2].set_xticklabels([c.replace("_", "\n") for c in cs], fontsize=7); ax[2].set_title("Mean P(fake) by condition")
    plt.tight_layout(); plt.savefig("ml/reports/nyaya.png", dpi=150)
    if os.environ.get("WANDB_API_KEY"):
        import wandb
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "satyacheck"), group="M5-Nyaya", name="nyaya-fusion", config={"alpha": alpha, "T": T})
        run.log({"test/auc": R["test"]["auc"], "test/coverage": R["test"]["coverage"], "test/abstain": R["test"]["abstain_rate"], "test/ece": R["test"]["ece"], "plot": wandb.Image("ml/reports/nyaya.png")}); run.finish()


if __name__ == "__main__":
    main()
