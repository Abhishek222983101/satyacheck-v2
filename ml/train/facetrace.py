"""M4 FaceTrace — blending / compositing artefacts in the face (blueprint §4.5).

EfficientNet-B0 (ImageNet init) on 224x224 aligned face crops. Trained ONLY on real faces + self-blended pseudo-fakes
(ml/data/sbi.py: face / lower-face / lips regions), then evaluated on held-out speakers. Real and fake share the same
random channel degradation (JPEG / resize / noise) so compression is not a shortcut.
Clip score = mean of the top-25% frame probabilities.  Grad-CAM lives in satya_core/explain (P8).
Run: python -m ml.train.facetrace --epochs 8         (W&B: source ~/.config/satya/wandb.env)"""
from __future__ import annotations

import argparse, math, os, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from ml.data.sbi import channel, make_fake

BANK = Path("data/face_bank")
MEAN, STD = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class FaceDataset(Dataset):
    def __init__(self, manifest, split, length, fixed_seed=False, seed=0, strength=None, region=None, augment=True):
        df = pd.read_parquet(manifest); df = df[df.split == split]
        self.ids = [c for c in df.clip_id if (BANK / f"{c}.npz").exists()]
        if not self.ids:
            raise RuntimeError(f"no face bank for split {split}")
        self.length, self.fixed, self.seed, self.strength, self.region, self.augment = length, fixed_seed, seed, strength, region, augment
        self._c: dict = {}

    def __len__(self):
        return self.length

    def _bank(self, cid):
        if cid not in self._c:
            z = np.load(BANK / f"{cid}.npz"); self._c[cid] = (z["faces"], z["lmk"].astype(np.float32))
            if len(self._c) > 128: self._c.pop(next(iter(self._c)))
        return self._c[cid]

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, i]) if self.fixed else np.random.default_rng()
        cid = self.ids[int(rng.integers(len(self.ids)))]; faces, lmk = self._bank(cid); k = int(rng.integers(len(faces)))
        img, lm = faces[k], lmk[k]; y = int(rng.random() < 0.5); reg = -1
        if y:
            img, r = make_fake(img, lm, rng, self.region); reg = ["face", "lower", "lips"].index(r)
        if self.augment and rng.random() < 0.5:
            img = img[:, ::-1]
        img = channel(np.ascontiguousarray(img), rng, self.strength)
        return torch.from_numpy(img).permute(2, 0, 1), torch.tensor(float(y)), torch.tensor(reg)


def build(pretrained=True):
    import timm
    m = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=1)
    return m


def prep(x):
    return ((x.float() / 255.0) - MEAN.to(x.device)) / STD.to(x.device)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay; self.keys = [k for k, v in model.state_dict().items() if v.dtype.is_floating_point]
        self.shadow = [model.state_dict()[k].detach().clone() for k in self.keys]

    @torch.no_grad()
    def update(self, model):
        sd = model.state_dict(); torch._foreach_lerp_(self.shadow, [sd[k].detach() for k in self.keys], 1 - self.decay)

    @torch.no_grad()
    def swap(self, model):
        sd = model.state_dict()
        for k, sh in zip(self.keys, self.shadow):
            t = sd[k].detach().clone(); sd[k].copy_(sh); sh.copy_(t)


@torch.no_grad()
def scores(model, ds, dev, bs=96, workers=4):
    model.eval(); S, Y, R = [], [], []
    for x, y, r in DataLoader(ds, batch_size=bs, num_workers=workers):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            S.append(torch.sigmoid(model(prep(x.to(dev))).float().squeeze(-1)).cpu()); Y.append(y); R.append(r)
    return torch.cat(S).numpy(), torch.cat(Y).numpy(), torch.cat(R).numpy()


def evaluate(model, manifest, split, dev, n=600):
    """AUC by channel severity (compression proxy for forwarding generations) and by blend region."""
    out = {}
    for name, st in (("clean", 0.05), ("mild", 0.35), ("heavy", 0.65), ("extreme", 0.95)):
        s, y, _ = scores(model, FaceDataset(manifest, split, n, True, 11, strength=st, augment=False), dev)
        out[f"auc_{name}"] = float(roc_auc_score(y, s)) if len(set(y)) > 1 else float("nan")
    for reg in ("face", "lower", "lips"):
        s, y, _ = scores(model, FaceDataset(manifest, split, n, True, 12, strength=0.35, region=reg, augment=False), dev)
        out[f"auc_{reg}"] = float(roc_auc_score(y, s)) if len(set(y)) > 1 else float("nan")
    out["auc_mean"] = float(np.nanmean([out[f"auc_{k}"] for k in ("clean", "mild", "heavy", "extreme")]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/video_real.parquet")
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--epoch-len", type=int, default=2400); ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--ema", type=float, default=0.99)
    ap.add_argument("--eval-n", type=int, default=600); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", default="ml/runs/facetrace")
    a = ap.parse_args(); dev = "cuda"; torch.manual_seed(0); np.random.seed(0)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tr = FaceDataset(a.manifest, "train", 96 if a.smoke else a.epoch_len)
    trl = DataLoader(tr, batch_size=a.bs, num_workers=2 if a.smoke else a.workers, drop_last=True, pin_memory=True, persistent_workers=not a.smoke, prefetch_factor=4)
    model = build().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4); total = a.epochs * len(trl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 60) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(total, 1))))))
    ema = EMA(model, a.ema)
    run = None
    if not a.no_wandb:
        import wandb
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "satyacheck"), group="M4-FaceTrace", name=f"facetrace-{time.strftime('%m%d-%H%M')}", config={**vars(a), "train_clips": len(tr.ids)})
    print(f"train clips {len(tr.ids)} | params {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    best, step = -1.0, 0
    for ep in range(a.epochs):
        model.train(); t0 = time.time()
        for x, y, _ in trl:
            x, y = prep(x.to(dev, non_blocking=True)), y.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.binary_cross_entropy_with_logits(model(x).float().squeeze(-1), y)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step(); sched.step(); ema.update(model); step += 1
            if step % 10 == 0:
                if run: run.log({"train/loss": loss.item(), "lr": opt.param_groups[0]["lr"], "epoch": ep}, step=step)
                print(f"ep{ep} step{step} loss {loss.item():.3f}", flush=True)
        ema.swap(model)
        m = evaluate(model, a.manifest, "train" if a.smoke else "val", dev, 96 if a.smoke else a.eval_n)
        print(f"== epoch {ep} ({time.time()-t0:.0f}s) auc_mean {m['auc_mean']:.4f} " + " ".join(f"{k[4:]} {v:.3f}" for k, v in m.items() if k != "auc_mean"), flush=True)
        if run: run.log({**{f"val/{k}": v for k, v in m.items()}}, step=step)
        if m["auc_mean"] > best:
            best = m["auc_mean"]; torch.save({"model": model.state_dict(), "metrics": m, "epoch": ep}, out / "best.pt"); print("   saved best", flush=True)
        ema.swap(model)
    if run: run.summary["best_auc_mean"] = best; run.finish()


if __name__ == "__main__":
    main()
