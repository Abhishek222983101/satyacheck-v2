"""M1 SyncFormer — audio-visual synchrony (lip-sync mismatch) — blueprint §4.3.

  mouth ROI (T,88,88) --VisualEnc--> v_t ─┐                        ┌─ Δ head: per-frame shift class (31 shifts + NOSYNC)
  log-mel  (4T,80)   --AudioEnc---> a_t ─┼─ CrossLayer x3 ─ fuse ─┤
  lip kinematics (T,8) ──────────────────┘   (+ kin added to v)   └─ F head (fake per frame; stage C)
  pre-fusion heads: contrastive projections (InfoNCE, ±15 frames) and teacher-distillation projections (SyncNet 1024-d)

Trained self-supervised on REAL clips only: the audio is shifted / swapped / partially replaced and the model must say
by how much (no fake generator involved). Sync score = 1 - P(|shift| <= 2) — natural clips carry ±2 frames of offset.
Run: python -m ml.train.syncformer --epochs 40      (W&B: source ~/.config/satya/wandb.env)
"""
from __future__ import annotations

import argparse, math, os, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

NCLS, ZERO, NOSYNC = 32, 15, 31


# ----------------------------------------------------------------------------- blocks
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(cout)
        self.sk = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)), inplace=True)
        return F.relu(self.b2(self.c2(y)) + self.sk(x), inplace=True)


class VisualEnc(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(1, 32, (5, 7, 7), (1, 2, 2), (2, 3, 3), bias=False), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
                                  nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1)))
        self.body = nn.Sequential(BasicBlock(32, 64, 2), BasicBlock(64, 128, 2), BasicBlock(128, 256, 2))
        self.proj = nn.Linear(256, d)

    def forward(self, x):                                   # (B,T,88,88) in [0,1]
        B, T = x.shape[:2]
        h = self.stem(x[:, None])                            # (B,32,T,22,22)
        h = h.permute(0, 2, 1, 3, 4).reshape(B * T, 32, h.shape[-2], h.shape[-1])
        h = self.body(h).mean((2, 3)).view(B, T, 256)
        return self.proj(h)


class SelfAttn(nn.Module):
    def __init__(self, d, h, p=0.1):
        super().__init__()
        self.n = nn.LayerNorm(d); self.a = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)

    def forward(self, x):
        y = self.n(x)
        return x + self.a(y, y, y, need_weights=False)[0]


class FFN(nn.Module):
    def __init__(self, d, p=0.1):
        super().__init__()
        self.n = nn.LayerNorm(d); self.f = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(p), nn.Linear(4 * d, d))

    def forward(self, x):
        return x + self.f(self.n(x))


class ConvMod(nn.Module):
    def __init__(self, d, k=15):
        super().__init__()
        self.n = nn.LayerNorm(d); self.pw = nn.Linear(d, 2 * d); self.dw = nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.bn = nn.BatchNorm1d(d); self.out = nn.Linear(d, d)

    def forward(self, x):
        y = F.glu(self.pw(self.n(x)), -1).transpose(1, 2)
        return x + self.out(F.silu(self.bn(self.dw(y))).transpose(1, 2))


class AudioEnc(nn.Module):
    def __init__(self, d, h=4):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(80, 128, 5, 1, 2), nn.BatchNorm1d(128), nn.GELU(),
                                  nn.Conv1d(128, d, 5, 2, 2), nn.BatchNorm1d(d), nn.GELU(),
                                  nn.Conv1d(d, d, 5, 2, 2), nn.BatchNorm1d(d), nn.GELU())     # 100 fps -> 25 fps
        self.blocks = nn.ModuleList([nn.Sequential(SelfAttn(d, h), ConvMod(d), FFN(d)) for _ in range(2)])

    def forward(self, mel):                                  # (B,4T,80)
        mel = (mel - mel.mean((1, 2), keepdim=True)) / (mel.std((1, 2), keepdim=True) + 1e-5)   # per-clip CMVN: gain-invariant
        x = self.conv(mel.transpose(1, 2)).transpose(1, 2)
        for b in self.blocks:
            x = b(x)
        return x


def sinusoid(T, d, device):
    pos = torch.arange(T, device=device)[:, None]; i = torch.arange(0, d, 2, device=device)[None]
    ang = pos / (10000 ** (i / d)); pe = torch.zeros(T, d, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(ang), torch.cos(ang)
    return pe


class CrossLayer(nn.Module):
    def __init__(self, d, h, p=0.1):
        super().__init__()
        self.sv, self.sa = SelfAttn(d, h, p), SelfAttn(d, h, p)
        self.nv, self.na = nn.LayerNorm(d), nn.LayerNorm(d)
        self.cv = nn.MultiheadAttention(d, h, dropout=p, batch_first=True); self.ca = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.fv, self.fa = FFN(d, p), FFN(d, p)

    def forward(self, v, a):
        v, a = self.sv(v), self.sa(a)
        nv, na = self.nv(v), self.na(a)
        v2 = v + self.cv(nv, na, na, need_weights=False)[0]
        a2 = a + self.ca(na, nv, nv, need_weights=False)[0]
        return self.fv(v2), self.fa(a2)


def cost_volume(pa, pv, maxs=15):
    """vol[b,t,k] = cos(audio_t, video_{t-s}) for s = k - maxs. Label convention: shift s>0 = audio delayed by s frames, i.e.
    the audio at frame t is original frame t-s, which pairs with video frame t-s. So argmax_k vol == s + maxs == the class index.
    pa, pv: L2-normalised (B,T,C). Out-of-range video frames contribute 0."""
    B, T, C = pv.shape
    pvp = F.pad(pv, (0, 0, maxs, maxs))                                   # (B, T+2*maxs, C)
    out = []
    for k in range(2 * maxs + 1):
        sh = k - maxs
        out.append((pa * pvp[:, maxs - sh: maxs - sh + T]).sum(-1))
    return torch.stack(out, -1)                                            # (B,T,31)


class SyncFormer(nn.Module):
    def __init__(self, d=192, heads=4, layers=3, kd_dim=1024, emb=128):
        super().__init__()
        self.d = d
        self.venc, self.aenc = VisualEnc(d), AudioEnc(d, heads)
        self.kin = nn.Sequential(nn.Linear(8, 64), nn.GELU(), nn.Linear(64, d))
        self.cross = nn.ModuleList([CrossLayer(d, heads) for _ in range(layers)])
        self.fuse = nn.Sequential(nn.Linear(2 * d, d), nn.GELU())
        self.delta = nn.Sequential(nn.Conv1d(d + 62, d, 5, padding=2), nn.GELU())      # fused features + 2 cost volumes (31 offsets each)
        self.delta_out = nn.Linear(d, NCLS)
        self.fake_out = nn.Linear(d, 1)
        self.pv, self.pa = nn.Linear(d, emb), nn.Linear(d, emb)          # contrastive projections (pre-fusion)
        self.kdv, self.kda = nn.Linear(d, kd_dim), nn.Linear(d, kd_dim)  # teacher-distillation projections (pre-fusion)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward(self, roi, mel, kin):
        """roi (B,T,88,88) uint8|float, mel (B,4T,80), kin (B,T,8) -> dict"""
        x = roi.float() / 255.0 if roi.dtype == torch.uint8 else roi
        ve, ae = self.venc(x), self.aenc(mel)
        T = min(ve.shape[1], ae.shape[1]); ve, ae = ve[:, :T], ae[:, :T]
        pe = sinusoid(T, self.d, ve.device)[None]
        v, a = ve + self.kin(kin[:, :T]) + pe, ae + pe
        for layer in self.cross:
            v, a = layer(v, a)
        f = self.fuse(torch.cat([v, a], -1))
        pvn, pan = F.normalize(self.pv(ve), dim=-1), F.normalize(self.pa(ae), dim=-1)
        kdv, kda = self.kdv(ve), self.kda(ae)
        vol_c, vol_k = cost_volume(pan, pvn), cost_volume(F.normalize(kda, dim=-1), F.normalize(kdv, dim=-1))
        vol = torch.cat([vol_c, vol_k], -1) * 5.0                                                                              # (B,T,62)
        f = self.delta(torch.cat([f, vol.to(f.dtype)], -1).transpose(1, 2)).transpose(1, 2)
        return {"delta": self.delta_out(f), "fake": self.fake_out(f).squeeze(-1), "ve": ve, "ae": ae,
                "pv": pvn, "pa": pan, "kdv": kdv, "kda": kda, "vol_c": vol_c, "vol_k": vol_k}


# ----------------------------------------------------------------------------- losses
def nce_loss(pv, pa, scale, band=15):
    """Frame-level InfoNCE inside a ±band neighbourhood; positive = same frame. pv,pa (B,T,C)."""
    T = pv.shape[1]
    sim = torch.einsum("btc,bsc->bts", pa, pv) * scale                                   # audio -> video
    idx = torch.arange(T, device=pv.device)
    mask = (idx[:, None] - idx[None]).abs() > band
    sim = sim.masked_fill(mask[None], -1e4)
    tgt = idx[None].expand(pv.shape[0], T)
    return 0.5 * (F.cross_entropy(sim.transpose(1, 2).reshape(-1, T), tgt.reshape(-1)) + F.cross_entropy(sim.transpose(1, 2).transpose(1, 2).reshape(-1, T), tgt.reshape(-1)))


def cos_kd(pred, tgt, m):
    c = 1 - F.cosine_similarity(pred.float(), tgt.float(), dim=-1)
    return (c * m).sum() / m.sum().clamp(min=1)


def vol_loss(vol, lbl, scale=20.0):
    """Direct offset supervision on a cost volume (B,T,31): treat scaled cosine similarities as logits over shifts.
    Frames whose true partner video frame (t - s) falls outside the crop carry no evidence and are skipped;
    NOSYNC frames are pushed toward a flat profile (no offset is consistent with unrelated speech)."""
    B, T, K = vol.shape
    t = torch.arange(T, device=vol.device)[None].expand(B, T)
    ok_shift = (lbl != NOSYNC)
    s = lbl - ZERO
    inb = ((t - s) >= 0) & ((t - s) < T)
    m = ok_shift & inb
    logits = (vol.float() * scale)
    ce = F.cross_entropy(logits[m], lbl[m]) if m.any() else vol.sum() * 0
    ns = (lbl == NOSYNC)
    flat = -F.log_softmax(logits[ns], -1).mean() if ns.any() else vol.sum() * 0          # cross-entropy to uniform
    return ce + 0.3 * (flat - math.log(K)).clamp(min=0)


def compute_loss(o, b, w_nce=0.5, w_kd=1.0, w_vol=1.0):
    T = o["delta"].shape[1]
    lbl = b["label"][:, :T]
    ce = F.cross_entropy(o["delta"].float().transpose(1, 2), lbl, label_smoothing=0.05)
    aligned = (b["kind"] == 0)
    l_nce = nce_loss(o["pv"][aligned].float(), o["pa"][aligned].float(), o["logit_scale"].exp().clamp(max=100)) if aligned.any() else ce * 0
    l_kd = 0.5 * (cos_kd(o["kdv"], b["kd_v"][:, :T], b["mv"][:, :T]) + cos_kd(o["kda"], b["kd_a"][:, :T], b["ma"][:, :T]))
    l_vol = vol_loss(o["vol_c"], lbl) + vol_loss(o["vol_k"], lbl)
    return ce + w_nce * l_nce + w_kd * l_kd + w_vol * l_vol, {"shift_ce": ce.item(), "nce": float(l_nce), "kd": float(l_kd), "vol": float(l_vol)}


# ----------------------------------------------------------------------------- EMA
class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.keys = [k for k, v in model.state_dict().items() if v.dtype.is_floating_point]
        self.shadow = [model.state_dict()[k].detach().clone() for k in self.keys]

    @torch.no_grad()
    def update(self, model):
        sd = model.state_dict()
        torch._foreach_lerp_(self.shadow, [sd[k].detach() for k in self.keys], 1 - self.decay)

    @torch.no_grad()
    def swap(self, model):
        sd = model.state_dict()
        for k, sh in zip(self.keys, self.shadow):
            t = sd[k].detach().clone(); sd[k].copy_(sh); sh.copy_(t)


# ----------------------------------------------------------------------------- evaluation
def sync_prob(delta):
    """delta logits (B,T,32) -> P(in sync) = P(|shift| <= 2)  (B,T)"""
    p = F.softmax(delta.float(), -1)
    return p[..., ZERO - 2: ZERO + 3].sum(-1)


def clip_score(sp, win=40):
    """High when ANY 1.6 s window is out of sync. sp (B,T) numpy."""
    k = np.ones(win) / win
    return np.array([1 - np.convolve(s, k, "valid").min() if len(s) >= win else 1 - s.mean() for s in sp])


@torch.no_grad()
def collect(model, ds, dev, bs=24, workers=4):
    model.eval(); R = {"sp": [], "lbl": [], "kind": [], "gen": [], "pred": []}
    for b in DataLoader(ds, batch_size=bs, num_workers=workers, shuffle=False):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(b["roi"].to(dev), b["mel"].to(dev), b["kin"].to(dev))
        R["sp"].append(sync_prob(o["delta"]).cpu().numpy()); R["pred"].append(o["delta"].float().argmax(-1).cpu().numpy())
        T = o["delta"].shape[1]; R["lbl"].append(b["label"][:, :T].numpy()); R["kind"] += b["kind"].tolist(); R["gen"] += b["gen"].tolist()
    return {k: (np.concatenate(v) if k in ("sp", "lbl", "pred") else np.array(v)) for k, v in R.items()}


def evaluate(model, mk_ds, dev, n=120):
    """mk_ds(kind, gen, seed) -> dataset. Returns metrics by fake type x generation (aligned clips are the negatives)."""
    out = {}
    for g in (0, 3, 5):
        neg = collect(model, mk_ds("aligned", g, 1000 + g), dev); s_neg = clip_score(neg["sp"])
        for kind in ("shifted", "swapped", "partial"):
            pos = collect(model, mk_ds(kind, g, 2000 + g), dev); s_pos = clip_score(pos["sp"])
            y = np.r_[np.zeros(len(s_neg)), np.ones(len(s_pos))]
            out[f"auc_{kind}_g{g}"] = float(roc_auc_score(y, np.r_[s_neg, s_pos]))
            if kind == "shifted":
                true = pos["lbl"][:, 0]; pred = np.array([np.bincount(r, minlength=NCLS).argmax() for r in pos["pred"]])
                out[f"shift_mae_frames_g{g}"] = float(np.abs(true - pred).mean()); out[f"shift_acc_g{g}"] = float((true == pred).mean())
            if kind == "partial":     # frame-level localisation of the replaced window
                fl = (pos["lbl"] != ZERO).ravel(); fs = (1 - pos["sp"]).ravel()
                out[f"loc_auc_partial_g{g}"] = float(roc_auc_score(fl, fs)) if fl.any() and (~fl).any() else float("nan")
    for k in ("shifted", "swapped", "partial"):
        out[f"auc_{k}"] = float(np.nanmean([out[f"auc_{k}_g{g}"] for g in (0, 3, 5)]))
    out["auc_mean"] = float(np.mean([out[f"auc_{k}"] for k in ("shifted", "swapped", "partial")]))
    out["auc_mean_g5"] = float(np.mean([out[f"auc_{k}_g5"] for k in ("shifted", "swapped", "partial")]))
    return out


# ----------------------------------------------------------------------------- train
def main():
    from ml.data.dataset_sync import SyncDataset
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/video_real.parquet")
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--epoch-len", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=24); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--ema", type=float, default=0.995)
    ap.add_argument("--w-kd", type=float, default=1.0); ap.add_argument("--w-vol", type=float, default=1.0); ap.add_argument("--w-nce", type=float, default=0.5)
    ap.add_argument("--eval-n", type=int, default=120); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", default="ml/runs/syncformer")
    a = ap.parse_args(); dev = "cuda"; torch.manual_seed(0); np.random.seed(0)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tr = SyncDataset(a.manifest, "train", 48 if a.smoke else a.epoch_len)
    mk_val = lambda kind, g, seed: SyncDataset(a.manifest, "train" if a.smoke else "val", 24 if a.smoke else a.eval_n, True, seed, force_gen=g, force_kind=kind, augment=False)
    trl = DataLoader(tr, batch_size=a.bs, num_workers=2 if a.smoke else a.workers, shuffle=False, drop_last=True, pin_memory=True, persistent_workers=not a.smoke, prefetch_factor=4)
    model = SyncFormer().to(dev)
    n_par = sum(p.numel() for p in model.parameters()); print(f"params {n_par/1e6:.2f}M | train clips {len(tr.ids)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    total = a.epochs * len(trl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(total, 1))))))
    ema = EMA(model, a.ema) if a.ema > 0 else None
    run = None
    if not a.no_wandb:
        import wandb
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "satyacheck"), group="M1-SyncFormer", name=f"syncformer-{time.strftime('%m%d-%H%M')}",
                         config={**vars(a), "params_M": n_par / 1e6, "train_clips": len(tr.ids)})
    best, step = -1.0, 0
    for ep in range(a.epochs):
        model.train(); t0 = time.time()
        for b in trl:
            bt = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(bt["roi"], bt["mel"], bt["kin"]); o["logit_scale"] = model.logit_scale
                loss, parts = compute_loss(o, bt, a.w_nce, a.w_kd, a.w_vol)
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1
            if ema: ema.update(model)
            if step % 10 == 0:
                if run: run.log({"train/loss": loss.item(), **{f"train/{k}": v for k, v in parts.items()}, "train/grad_norm": float(gn), "lr": opt.param_groups[0]["lr"], "epoch": ep}, step=step)
                print(f"ep{ep} step{step} loss {loss.item():.3f} " + " ".join(f"{k} {v:.3f}" for k, v in parts.items()), flush=True)
        if ema: ema.swap(model)
        m = evaluate(model, mk_val, dev)
        score = m["auc_mean"] * 0.5 + m["auc_mean_g5"] * 0.5
        print(f"== epoch {ep} ({time.time()-t0:.0f}s) score {score:.4f} auc_mean {m['auc_mean']:.3f} g5 {m['auc_mean_g5']:.3f} | shifted {m['auc_shifted']:.3f} swapped {m['auc_swapped']:.3f} partial {m['auc_partial']:.3f} | shift_acc_g0 {m['shift_acc_g0']:.2f} mae {m['shift_mae_frames_g0']:.1f}fr", flush=True)
        if run: run.log({**{f"val/{k}": v for k, v in m.items()}, "val/score": score}, step=step)
        if score > best:
            best = score; torch.save({"model": model.state_dict(), "metrics": m, "epoch": ep, "args": vars(a)}, out / "best.pt"); print("   saved best", flush=True)
        if ema: ema.swap(model)
    if run:
        run.summary["best_score"] = best; run.finish()


if __name__ == "__main__":
    main()
