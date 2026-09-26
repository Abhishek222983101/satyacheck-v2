"""M2 SpliceLoc — audio splice localisation + synthetic-voice detection (blueprint §4.4).

XLS-R-300M truncated to its first 8 transformer layers (lower/middle SSL layers carry most anti-spoofing evidence;
NAACL Findings 2025) -> learnable weighted sum of layers 4..8 -> 256-d -> BAM-style heads:
  * Boundary Enhancement: intra-frame conv branch + inter-frame self-attention -> boundary logit / 20 ms frame
  * Boundary-aware Frame Attention x2: self-attention where a frame may only attend to frames in its own segment
    (segments are cut at predicted boundaries; ground-truth boundaries used with p=0.5 during training)
  * frame spoof head (2-way), utterance head -> [p_synthetic, p_spliced]

Run:  python -m ml.train.spliceloc --epochs 10          (W&B: source ~/.config/satya/wandb.env)
"""
from __future__ import annotations

import argparse, math, os, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

XLSR = os.path.expanduser("~/.cache/satya/weights/xlsr300m")
FRAME_S = 0.02


# ----------------------------------------------------------------------------- model
def run_starts(bin_: torch.Tensor) -> torch.Tensor:
    """(B,T) {0,1} -> 1 where a run of ones begins."""
    prev = F.pad(bin_[:, :-1], (1, 0))
    return ((bin_ > 0.5) & (prev < 0.5)).float()


class MaskedAttnBlock(nn.Module):
    def __init__(self, d: int, heads: int, p: float = 0.1):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(p), nn.Linear(4 * d, d))

    def forward(self, x, seg):
        B, T, _ = x.shape
        allow = seg[:, :, None] == seg[:, None, :]                      # (B,T,T) same segment (invalid frames = seg -1)
        mask = (~allow).repeat_interleave(self.h, 0)                    # True = blocked, (B*h,T,T); diagonal always open
        y = self.n1(x)
        x = x + self.att(y, y, y, attn_mask=mask, need_weights=False)[0]
        return x + self.ff(self.n2(x))


class SpliceLoc(nn.Module):
    def __init__(self, ssl_path: str = XLSR, d: int = 256, heads: int = 4, n_layers: int = 8, use_layers=(4, 5, 6, 7, 8)):
        super().__init__()
        from transformers import Wav2Vec2Config, Wav2Vec2Model
        cfg = Wav2Vec2Config.from_pretrained(ssl_path)
        cfg.num_hidden_layers, cfg.layerdrop = n_layers, 0.0
        cfg.apply_spec_augment, cfg.mask_time_prob, cfg.mask_feature_prob = False, 0.0, 0.0
        self.ssl = Wav2Vec2Model.from_pretrained(ssl_path, config=cfg)
        self.ssl.freeze_feature_encoder()
        self.use = list(use_layers)
        self.lw = nn.Parameter(torch.zeros(len(self.use)))
        self.proj = nn.Sequential(nn.LayerNorm(cfg.hidden_size), nn.Linear(cfg.hidden_size, d))
        # boundary enhancement
        self.intra = nn.Sequential(nn.Conv1d(d, d, 3, padding=1), nn.GroupNorm(1, d), nn.GELU(), nn.Conv1d(d, d, 3, padding=1))
        self.inter = nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.bnd_head = nn.Linear(2 * d, 1)
        # boundary-aware frame attention
        self.bfa = nn.ModuleList([MaskedAttnBlock(d, heads) for _ in range(2)])
        self.frame_head = nn.Linear(d, 2)
        self.att_pool = nn.Linear(d, 1)
        self.utt = nn.Sequential(nn.Linear(2 * d + 2, d), nn.GELU(), nn.Dropout(0.2), nn.Linear(d, 2))
        self.ssl_frozen = False

    def set_ssl_trainable(self, first_trainable_layer: int | None):
        """None = freeze all SSL; k = train encoder layers >= k (0-indexed) only."""
        for p in self.ssl.parameters():
            p.requires_grad = False
        self.ssl_frozen = first_trainable_layer is None
        if first_trainable_layer is not None:
            for i, layer in enumerate(self.ssl.encoder.layers):
                if i >= first_trainable_layer:
                    for p in layer.parameters():
                        p.requires_grad = True
            for p in self.ssl.encoder.layer_norm.parameters():
                p.requires_grad = True

    def features(self, wav, n_valid_samples):
        # per-utterance normalisation over the valid region only (padding would skew mean/std)
        idx = torch.arange(wav.shape[1], device=wav.device)[None]
        m = (idx < n_valid_samples[:, None]).float()
        mean = (wav * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1)
        var = (((wav - mean) * m) ** 2).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1)
        x = (wav - mean) / torch.sqrt(var + 1e-7) * m
        ctx = torch.no_grad() if self.ssl_frozen else torch.enable_grad()
        with ctx:
            hs = self.ssl(x, output_hidden_states=True).hidden_states
        st = torch.stack([F.layer_norm(hs[i], hs[i].shape[-1:]) for i in self.use], 0)      # (L,B,T,1024)
        w = torch.softmax(self.lw, 0)[:, None, None, None]
        return self.proj((st * w).sum(0))                                                    # (B,T,d)

    def forward(self, wav, n_valid_samples, gt_boundary=None, valid=None, teacher_p: float = 0.5):
        h = self.features(wav, n_valid_samples)
        B, T, _ = h.shape
        if valid is None:
            valid = torch.ones(B, T, device=h.device)
        valid = valid[:, :T]
        # --- boundary enhancement
        a = h + self.intra(h.transpose(1, 2)).transpose(1, 2)
        b = self.inter(h)
        bl = self.bnd_head(torch.cat([a, b], -1)).squeeze(-1)                                 # (B,T) logits
        # --- segment ids from boundaries (GT with prob teacher_p during training, else predicted)
        with torch.no_grad():
            pred_bin = (torch.sigmoid(bl.float()) > 0.5).float()
            if gt_boundary is not None and self.training:
                use_gt = (torch.rand(B, 1, device=h.device) < teacher_p).float()
                bin_ = use_gt * gt_boundary[:, :T] + (1 - use_gt) * pred_bin
            else:
                bin_ = pred_bin
            seg = torch.cumsum(run_starts(bin_ * valid), 1).long()
            seg = torch.where(valid > 0.5, seg, torch.full_like(seg, -1))
        x = a + b
        for blk in self.bfa:
            x = blk(x, seg)
        fl = self.frame_head(x)                                                               # (B,T,2)
        # --- utterance head: attentive stats pooling + boundary/spoof summaries
        s = self.att_pool(x).squeeze(-1).masked_fill(valid < 0.5, -1e4)
        wgt = torch.softmax(s, 1)[..., None]
        mu = (wgt * x).sum(1)
        sd = torch.sqrt((wgt * (x - mu[:, None]) ** 2).sum(1) + 1e-5)
        pb = torch.sigmoid(bl.float()).masked_fill(valid < 0.5, 0).amax(1, keepdim=True)
        ps = (torch.softmax(fl.float(), -1)[..., 1] * valid).sum(1, keepdim=True) / valid.sum(1, keepdim=True).clamp(min=1)
        ul = self.utt(torch.cat([mu, sd, pb.to(mu.dtype), ps.to(mu.dtype)], -1))              # (B,2) [synth, splice]
        return {"boundary": bl, "frame": fl, "utt": ul, "valid": valid}


def compute_loss(out, batch, pos_w: float = 8.0):
    T = out["boundary"].shape[1]
    v = out["valid"]
    fs = batch["frame_spoof"][:, :T]
    ce = F.cross_entropy(out["frame"].float().transpose(1, 2), fs, weight=torch.tensor([1.0, 2.0], device=v.device), reduction="none")
    l_frame = (ce * v).sum() / v.sum().clamp(min=1)
    bce = F.binary_cross_entropy_with_logits(out["boundary"].float(), batch["boundary"][:, :T], pos_weight=torch.tensor(pos_w, device=v.device), reduction="none")
    l_bnd = (bce * v).sum() / v.sum().clamp(min=1)
    tgt = torch.stack([batch["utt_synth"], batch["utt_splice"]], 1)
    l_utt = F.binary_cross_entropy_with_logits(out["utt"].float(), tgt)
    total = l_frame + 0.5 * l_bnd + 0.5 * l_utt
    return total, {"frame": l_frame.item(), "boundary": l_bnd.item(), "utt": l_utt.item()}


# ----------------------------------------------------------------------------- metrics
def eer(y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    i = np.nanargmin(np.abs(fpr - (1 - tpr)))
    return float((fpr[i] + 1 - tpr[i]) / 2)


def pick_peaks(p: np.ndarray, thr: float, min_gap: int = 15) -> list[int]:
    idx = [i for i in range(1, len(p) - 1) if p[i] >= thr and p[i] >= p[i - 1] and p[i] >= p[i + 1]]
    idx.sort(key=lambda i: -p[i])
    keep: list[int] = []
    for i in idx:
        if all(abs(i - j) >= min_gap for j in keep):
            keep.append(i)
    return sorted(keep)


def gt_centres(bd: np.ndarray) -> list[int]:
    idx, out, s = np.nonzero(bd > 0.5)[0], [], None
    if len(idx) == 0:
        return out
    run = [idx[0]]
    for i in idx[1:]:
        if i == run[-1] + 1:
            run.append(i)
        else:
            out.append(int(np.mean(run))); run = [i]
    out.append(int(np.mean(run)))
    return out


def boundary_f1(preds, gts, tol=4):
    tp = fp = fn = 0
    for p, g in zip(preds, gts):
        used = set()
        for pi in p:
            m = [j for j, gj in enumerate(g) if j not in used and abs(gj - pi) <= tol]
            if m:
                used.add(min(m, key=lambda j: abs(g[j] - pi))); tp += 1
            else:
                fp += 1
        fn += len(g) - len(used)
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return 2 * prec * rec / max(prec + rec, 1e-9), prec, rec


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    model.eval()
    fy, fs, bp, bg, uy, us, gens = [], [], [], [], [], [], []
    for bi, b in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        wav = b["wav"].to(device)
        nvs = (b["valid"].sum(1) * 320).long().to(device).clamp(max=wav.shape[1])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(wav, nvs, valid=b["valid"].to(device))
        T = o["boundary"].shape[1]
        prob_sp = torch.softmax(o["frame"].float(), -1)[..., 1].cpu().numpy()
        prob_b = torch.sigmoid(o["boundary"].float()).cpu().numpy()
        v = b["valid"][:, :T].numpy() > 0.5
        for i in range(len(wav)):
            fy.append(b["frame_spoof"][i, :T].numpy()[v[i]]); fs.append(prob_sp[i][v[i]])
            bp.append(prob_b[i] * v[i]); bg.append(gt_centres(b["boundary"][i, :T].numpy()))
        uy.append(torch.stack([b["utt_synth"], b["utt_splice"]], 1).numpy()); us.append(torch.sigmoid(o["utt"].float()).cpu().numpy())
        gens.append(b["gen"].numpy())
    fy, fs = np.concatenate(fy), np.concatenate(fs)
    uy, us, gens = np.concatenate(uy), np.concatenate(us), np.concatenate(gens)
    best = max(((boundary_f1([pick_peaks(p, t) for p in bp], bg), t) for t in (0.3, 0.4, 0.5, 0.6, 0.7)), key=lambda z: z[0][0])
    (f1, pr, rc), thr = best
    m = {"frame_eer": eer(fy, fs), "boundary_f1": f1, "boundary_prec": pr, "boundary_rec": rc, "boundary_thr": thr}
    for k, name in enumerate(("synth", "splice")):
        m[f"auc_{name}"] = roc_auc_score(uy[:, k], us[:, k]) if len(np.unique(uy[:, k])) > 1 else float("nan")
        for lo, hi, tag in ((0, 1, "g01"), (2, 3, "g23"), (4, 5, "g45")):
            sel = (gens >= lo) & (gens <= hi)
            m[f"auc_{name}_{tag}"] = roc_auc_score(uy[sel, k], us[sel, k]) if sel.sum() > 20 and len(np.unique(uy[sel, k])) > 1 else float("nan")
    return m, (bp, bg)


# ----------------------------------------------------------------------------- EMA
class EMA:
    """Exponential moving average of all float weights; evaluation/checkpoints use the averaged weights, which
    removes most of the epoch-to-epoch validation bounce seen with batch-12 SGD noise."""
    def __init__(self, model, decay: float):
        self.decay = decay
        self.keys = [k for k, v in model.state_dict().items() if v.dtype.is_floating_point]
        self.shadow = [model.state_dict()[k].detach().clone() for k in self.keys]

    @torch.no_grad()
    def update(self, model):
        sd = model.state_dict()
        torch._foreach_lerp_(self.shadow, [sd[k].detach() for k in self.keys], 1 - self.decay)

    @torch.no_grad()
    def swap(self, model):
        """Exchange live and averaged weights in place (call twice to restore)."""
        sd = model.state_dict()
        for k, sh in zip(self.keys, self.shadow):
            t = sd[k].detach().clone(); sd[k].copy_(sh); sh.copy_(t)


# ----------------------------------------------------------------------------- train
def main():
    from ml.data.dataset_splice import SpliceDataset
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/audio_split.parquet")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--epoch-len", type=int, default=3600)
    ap.add_argument("--val-len", type=int, default=600)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--warm-epochs", type=int, default=1)
    ap.add_argument("--unfreeze-from", type=int, default=4)
    ap.add_argument("--lr-head", type=float, default=5e-4)
    ap.add_argument("--lr-ssl", type=float, default=1e-5)
    ap.add_argument("--ema", type=float, default=0.995)
    ap.add_argument("--tf-start", type=float, default=0.5, help="P(use GT boundaries for attention masks) at epoch 0")
    ap.add_argument("--tf-end", type=float, default=0.5, help="... at the last epoch (scheduled sampling: lower = train on own predicted cuts)")
    ap.add_argument("--manifest-tag", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", default="ml/runs/spliceloc")
    a = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    dev = "cuda"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    tr = SpliceDataset(a.manifest, "train", a.epoch_len if not a.smoke else 24)
    va = SpliceDataset(a.manifest, "val", a.val_len if not a.smoke else 16, fixed_seed=True, seed=123)
    kw = dict(batch_size=a.bs, num_workers=a.workers if not a.smoke else 2, pin_memory=True, persistent_workers=not a.smoke, prefetch_factor=4)
    trl, val = DataLoader(tr, shuffle=False, drop_last=True, **kw), DataLoader(va, shuffle=False, **kw)

    model = SpliceLoc().to(dev)
    model.ssl.gradient_checkpointing_enable()
    head_params = [p for n, p in model.named_parameters() if not n.startswith("ssl.")]
    ssl_params = [p for n, p in model.named_parameters() if n.startswith("ssl.")]
    opt = torch.optim.AdamW([{"params": head_params, "lr": a.lr_head}, {"params": ssl_params, "lr": a.lr_ssl}], weight_decay=0.01)
    ema = EMA(model, a.ema) if a.ema > 0 else None
    steps_total = a.epochs * (len(trl))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(steps_total, 1)))))

    run = None
    if not a.no_wandb:
        import wandb
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "satyacheck"), group="M2-SpliceLoc", name=f"spliceloc-{time.strftime('%m%d-%H%M')}",
                         config={**vars(a), "params_total_M": sum(p.numel() for p in model.parameters()) / 1e6,
                                 "params_head_M": sum(p.numel() for p in head_params) / 1e6, "ssl_layers": 8, "gen_p": [0.25, 0.20, 0.20, 0.15, 0.10, 0.10]})
    print(f"params: total {sum(p.numel() for p in model.parameters())/1e6:.1f}M, heads {sum(p.numel() for p in head_params)/1e6:.1f}M", flush=True)

    best, step = -1.0, 0
    for ep in range(a.epochs):
        model.set_ssl_trainable(None if ep < a.warm_epochs else a.unfreeze_from)
        model.train(); t0 = time.time(); agg = {}
        for b in trl:
            wav = b["wav"].to(dev, non_blocking=True)
            bt = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            nvs = (bt["valid"].sum(1) * 320).long().clamp(max=wav.shape[1])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(wav, nvs, gt_boundary=bt["boundary"], valid=bt["valid"], teacher_p=a.tf_start + (a.tf_end - a.tf_start) * ep / max(1, a.epochs - 1))
                loss, parts = compute_loss(o, bt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if ema is not None: ema.update(model)
            for k, v in parts.items():
                agg[k] = agg.get(k, 0) + v
            if step % 10 == 0:
                msg = {"train/loss": loss.item(), **{f"train/{k}": v for k, v in parts.items()}, "train/grad_norm": float(gn),
                       "lr_head": opt.param_groups[0]["lr"], "lr_ssl": opt.param_groups[1]["lr"], "epoch": ep}
                if run: run.log(msg, step=step)
                print(f"ep{ep} step{step} loss {loss.item():.3f} " + " ".join(f"{k} {v:.3f}" for k, v in parts.items()), flush=True)
        if ema is not None: ema.swap(model)
        m, (bp, bg) = evaluate(model, val, dev)
        # selection follows the problem statement (splicing): splice detection + cut localisation + synthetic voice.
        # frame_eer is logged but excluded: it mixes whole-clip-synthetic labels with local-insertion labels.
        score = np.nanmean([m["auc_synth"], m["auc_splice"], m["boundary_f1"]])
        print(f"== epoch {ep} ({time.time()-t0:.0f}s) score {score:.4f} " + " ".join(f"{k} {v:.3f}" for k, v in m.items()), flush=True)
        if run:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, axs = plt.subplots(3, 1, figsize=(9, 5), sharex=True)
            for ax, i in zip(axs, range(3)):
                ax.plot(bp[i], color="#C2338F", lw=1.2, label="P(boundary)")
                for g in bg[i]: ax.axvline(g, color="k", ls="--", lw=1)
                ax.set_ylim(0, 1)
            axs[0].legend(loc="upper right"); axs[-1].set_xlabel("frame (20 ms)"); fig.suptitle(f"val epoch {ep}: dashed = true cut")
            run.log({**{f"val/{k}": v for k, v in m.items()}, "val/score": score, "val/examples": wandb.Image(fig)}, step=step); plt.close(fig)
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "metrics": m, "epoch": ep, "args": vars(a), "ema": a.ema}, out / "best.pt")
            print(f"   saved best ({score:.4f})", flush=True)
        if ema is not None: ema.swap(model)          # restore live weights before training continues
    if run:
        run.summary["best_score"] = best; run.finish()


if __name__ == "__main__":
    main()
