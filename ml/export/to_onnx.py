"""P8.1 — export M1 (SyncFormer), M4 (FaceTrace) and M5 (Nyaya) to ONNX + INT8 + parity,
and assemble models/registry.json (hashes, input specs, calib) per blueprint §4.12.

Fixed 10 s window for M1 (T=250 @25 fps, mel=1000): dynamo export fails on cost-volume
shape arithmetic, so we trace the static window and let the runtime pad/slice (the same
pattern ml/export/spliceloc_onnx.py uses for its 160000-sample window).

Usage:
  python -m ml.export.to_onnx                 # export + int8 + parity + registry
  python -m ml.export.to_onnx --parity-n 40    # parity on N val clips (default 40)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

M1_T, MEL_T, M1_WIN = 250, 1000, 88          # 10 s @ 25 fps
REG = Path("models/registry.json")


# ---------------------------------------------------------------- wrappers
class M1Infer(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, roi, mel, kin):
        o = self.m(roi, mel, kin)
        return o["delta"], o["fake"]


class FTInfer(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x):                     # already normalised 0..1 RGB (B,3,224,224)
        return self.m(x)


class NyayaInfer(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, S, M, C):
        z, g = self.m(S, M, C)
        return z, g


def sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def export(net, args, inputs, outputs, out, dynamic=None, opset=18):
    t0 = time.time()
    torch.onnx.export(net, args, out, input_names=inputs, output_names=outputs,
                      opset_version=opset, dynamo=False, dynamic_axes=dynamic or {})
    print(f"  exported {out} ({Path(out).stat().st_size / 1e6:.0f} MB) in {time.time() - t0:.0f}s")


def ort_run(path, feeds):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    return s.run(None, feeds)


def parity(path, torch_fn, feeds, label, tol=1e-2):
    """max |pytorch - ORT| across outputs; blueprint gate: fp32 < 1e-2."""
    with torch.no_grad():
        outs = torch_fn(feeds)
    got = ort_run(path, feeds)
    diffs = [float(np.abs(o - np.asarray(r)).max()) for o, r in zip(outs, got)]
    mx = max(diffs)
    flag = "OK " if mx < tol else "FAIL"
    print(f"  {flag} parity {label}: max|diff| {mx:.5f} (limit {tol})")
    return mx


def quantize(src: str, dst: str):
    import onnx
    from onnxsim import simplify
    from onnxruntime.quantization import QuantType, quantize_dynamic
    m, ok = simplify(onnx.load(src))
    assert ok, "onnxsim not ok"
    sim = src.replace(".onnx", ".sim.onnx")
    onnx.save(m, sim)
    quantize_dynamic(sim, dst, weight_type=QuantType.QInt8, per_channel=True)
    Path(sim).unlink(missing_ok=True)
    print(f"  int8 -> {dst} ({Path(dst).stat().st_size / 1e6:.0f} MB)")


# ---------------------------------------------------------------- parity on real clips
def m1_real_parity(m1: torch.nn.Module, path: str, n: int, tol: float = 1e-2) -> float:
    """M1 fp32/int8 parity on up to n cached val clips (data/cache_v), padded to the fixed
    10 s window; compares only the first T valid frames of each output."""
    import glob
    clips = sorted(glob.glob("data/cache_v/*/g0.npz"))[:n]
    if not clips:
        print("  (no cached clips in data/cache_v — skipping real-clip parity)")
        return 0.0
    worst, seen = 0.0, 0
    for c in clips:
        z = np.load(c); roi, mel = z["roi"], z["mel"]
        kin_p = np.load(str(Path(c).parent / "kin.npy"))
        T = min(len(roi), len(kin_p), mel.shape[0] // 4)
        if T < 50:
            continue
        roi_p = np.zeros((M1_T, M1_WIN, M1_WIN), np.uint8); roi_p[:T] = roi[:T]
        mel_p = np.zeros((MEL_T, 80), np.float32); mel_p[:4 * T] = mel[:4 * T]
        kin_p2 = np.zeros((M1_T, 8), np.float32); kin_p2[:T] = kin_p[:T]
        feeds = {"roi": roi_p[None], "mel": mel_p[None], "kin": kin_p2[None]}
        got = ort_run(path, feeds)
        with torch.no_grad():
            o = m1(torch.from_numpy(feeds["roi"]), torch.from_numpy(feeds["mel"]), torch.from_numpy(feeds["kin"]))
        refs = [o["delta"].numpy()[:, :T], o["fake"].numpy()[:, :T]]
        d = max(float(np.abs(g - r).max()) for g, r in zip([got[0][:, :T], got[1][:, :T]], refs))
        worst = max(worst, d); seen += 1
    flag = "OK " if worst < tol else "FAIL"
    print(f"  {flag} parity M1 on {seen} val clips ({path}): max|diff| {worst:.5f}")
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity-n", type=int, default=40)
    ap.add_argument("--skip-int8", action="store_true")
    a = ap.parse_args()
    Path("models").mkdir(exist_ok=True)
    torch.set_num_threads(4)

    # ---- M1 SyncFormer -------------------------------------------------
    print("M1 syncformer")
    from ml.train.syncformer import SyncFormer
    ck = torch.load("ml/runs/syncformer_best.pt", map_location="cpu", weights_only=True)
    m1 = SyncFormer(); m1.load_state_dict(ck["model"]); m1.eval()
    net = M1Infer(m1).eval()
    roi = (torch.rand(1, M1_T, M1_WIN, M1_WIN) * 255).to(torch.uint8)
    mel, kin = torch.rand(1, MEL_T, 80), torch.rand(1, M1_T, 8)
    with torch.no_grad():
        ref = [t.numpy() for t in net(roi, mel, kin)]
    export(net, (roi, mel, kin), ["roi", "mel", "kin"], ["delta", "fake"],
           "models/syncformer.onnx", dynamic={"roi": {0: "B"}, "mel": {0: "B"}, "kin": {0: "B"},
                                              "delta": {0: "B"}, "fake": {0: "B"}})
    parity("models/syncformer.onnx", lambda f: [t.numpy() for t in net(torch.from_numpy(f["roi"]), torch.from_numpy(f["mel"]), torch.from_numpy(f["kin"]))],
           {"roi": roi.numpy(), "mel": mel.numpy(), "kin": kin.numpy()}, "M1 fp32")
    m1_real_parity(m1, "models/syncformer.onnx", a.parity_n)
    if not a.skip_int8:
        quantize("models/syncformer.onnx", "models/syncformer.int8.onnx")
        parity("models/syncformer.int8.onnx", lambda f: [t.numpy() for t in net(torch.from_numpy(f["roi"]), torch.from_numpy(f["mel"]), torch.from_numpy(f["kin"]))],
               {"roi": roi.numpy(), "mel": mel.numpy(), "kin": kin.numpy()}, "M1 int8", tol=5e-2)
        m1_real_parity(m1, "models/syncformer.int8.onnx", a.parity_n, tol=5e-2)

    # ---- M4 FaceTrace --------------------------------------------------
    print("M4 facetrace")
    from ml.train.facetrace import build
    ck = torch.load("ml/runs/facetrace_v2/best.pt", map_location="cpu", weights_only=True)
    m4 = build(pretrained=False); m4.load_state_dict(ck["model"]); m4.eval()
    net4 = FTInfer(m4).eval()
    x = torch.rand(4, 3, 224, 224)
    export(net4, (x,), ["x"], ["p"], "models/facetrace.onnx", dynamic={"x": {0: "B"}, "p": {0: "B"}})
    parity("models/facetrace.onnx", lambda f: [m4(torch.from_numpy(f["x"])).numpy()],
           {"x": x.numpy()}, "M4 fp32")
    if not a.skip_int8:
        quantize("models/facetrace.onnx", "models/facetrace.int8.onnx")
        parity("models/facetrace.int8.onnx", lambda f: [m4(torch.from_numpy(f["x"])).numpy()],
               {"x": x.numpy()}, "M4 int8", tol=5e-2)

    # ---- M5 Nyaya (tiny MLP, fp32 only) --------------------------------
    print("M5 nyaya")
    from ml.train.nyaya import Nyaya
    ck = torch.load("models/nyaya.pt", map_location="cpu", weights_only=True)
    ny = Nyaya(ck.get("mode", "mean")); ny.load_state_dict(ck["model"]); ny.eval()
    net5 = NyayaInfer(ny).eval()
    S, M, C = torch.randn(4, 4), torch.ones(4, 4), torch.rand(4, 3)
    export(net5, (S, M, C), ["S", "M", "C"], ["z", "g"], "models/nyaya.onnx",
           dynamic={"S": {0: "B"}, "M": {0: "B"}, "C": {0: "B"}, "z": {0: "B"}, "g": {0: "B"}})
    parity("models/nyaya.onnx", lambda f: [t.numpy() for t in net5(torch.from_numpy(f["S"]), torch.from_numpy(f["M"]), torch.from_numpy(f["C"]))],
           {"S": S.numpy(), "M": M.numpy(), "C": C.numpy()}, "M5 fp32")

    # ---- registry -------------------------------------------------------
    print("INT8 AUC gate (keep int8 only if delta <= 0.5 pt)…")
    _, keep = auc_gate()
    bench = Path("ml/reports/benchmark.json")
    reg = {
        "generated_by": "python -m ml.export.to_onnx",
        "opset": 18,
        "artifacts": {
            "syncformer": {"version": "v1-epoch15", "onnx_path": "models/syncformer.onnx", "sha256": sha("models/syncformer.onnx"),
                           "int8_path": "models/syncformer.int8.onnx", "sha256_int8": sha("models/syncformer.int8.onnx"),
                           "input_spec": {"roi": ["B", 250, 88, 88], "mel": ["B", 1000, 80], "kin": ["B", 250, 8]},
                           "outputs": ["delta (B,T,32)", "fake (B,T)"], "trained_on": "TalkVid-Bench TalkVid-1199, speaker-disjoint"},
            "facetrace": {"version": "v2", "onnx_path": "models/facetrace.onnx", "sha256": sha("models/facetrace.onnx"),
                          "int8_path": "models/facetrace.int8.onnx", "sha256_int8": sha("models/facetrace.int8.onnx"),
                          "input_spec": {"x": ["B", 3, 224, 224]}, "outputs": ["p (B)"],
                          "trained_on": "self-blended pseudo-fakes on real faces, channel-shared"},
            "spliceloc": {"version": "v2", "onnx_path": "models/spliceloc.onnx", "sha256": sha("models/spliceloc.onnx"),
                          "int8_path": "models/spliceloc.int8.onnx",
                          "sha256_int8": sha("models/spliceloc.int8.onnx") if Path("models/spliceloc.int8.onnx").exists() else None,
                          "input_spec": {"wav": ["B", 160000], "n_valid": ["B"]},
                          "outputs": ["boundary (B,T)", "frame (B,T,2)", "utt (B,2)"], "trained_on": "splice factory on-the-fly, IndicTTS/FLEURS audio"},
            "nyaya": {"version": "nyaya-fit", "onnx_path": "models/nyaya.onnx", "sha256": sha("models/nyaya.onnx"),
                      "input_spec": {"S": ["B", 4], "M": ["B", 4], "C": ["B", 3]}, "outputs": ["z (B)", "g (B,4)"],
                      "calib": {"T": float(ck["T"]), "qhat_by_bucket": ck["qhat"], "alpha": float(ck["alpha"])},
                      "trained_on": "nyaya_{val,cal,test} speaker-disjoint splits"},
        },
        "metrics_ref": str(bench) if bench.exists() else None,
        "serving": {"syncformer": keep.get("M1", "fp32"), "facetrace": keep.get("M4", "fp32"),
                    "spliceloc": "int8", "nyaya": "fp32"},
    }
    REG.write_text(json.dumps(reg, indent=2))
    print("registry ->", REG)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- INT8 AUC gate (§4.12)
def _m1_scores(path, clips):
    """clip score = mean p_nosync from the delta head, gens present in cache."""
    import glob
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    out = []
    for c in clips:
        z = np.load(c); roi, mel = z["roi"], z["mel"]
        kin_p = np.load(str(Path(c).parent / "kin.npy"))
        T = min(len(roi), len(kin_p), mel.shape[0] // 4)
        if T < 50:
            continue
        roi_p = np.zeros((M1_T, M1_WIN, M1_WIN), np.uint8); roi_p[:T] = roi[:T]
        mel_p = np.zeros((MEL_T, 80), np.float32); mel_p[:4 * T] = mel[:4 * T]
        kin_p2 = np.zeros((M1_T, 8), np.float32); kin_p2[:T] = kin_p[:T]
        delta, _ = s.run(None, {"roi": roi_p[None], "mel": mel_p[None], "kin": kin_p2[None]})
        from scipy.special import softmax
        from ml.train.syncformer import NOSYNC
        cm = softmax(delta[0, :T], -1).mean(0)          # class mean over valid frames
        out.append(float(cm[NOSYNC]))
    return out


def auc_gate():
    """Compare fp32 vs int8 AUC; blueprint: keep fp32 if INT8 drifts > 0.5 AUC points."""
    import glob
    from sklearn.metrics import roc_auc_score
    res = {}
    # M1: discriminate heavy vs light compression on the cache ladder (ranking-stability task)
    clips = sorted(glob.glob("data/cache_v/*/g0.npz"))[:30]
    if clips:
        y = []
        # label by folder-independent proxy: use ladder gen from meta (gen>=4 vs <=1) on g0 (clean) is not possible;
        # instead use cached re-encoded ladders g4/g5 vs g0/g1 of the SAME clips
        X, Y = [], []
        for c in clips:
            for g, lab in ((0, 0), (1, 0), (4, 1), (5, 1)):
                p = c.replace("g0.npz", f"g{g}.npz")
                if Path(p).exists():
                    X.append(p); Y.append(lab)
        from sklearn.metrics import roc_auc_score as ra
        s32 = _m1_scores("models/syncformer.onnx", X)
        s8 = _m1_scores("models/syncformer.int8.onnx", X)
        res["M1"] = {"n": len(X), "auc_fp32": ra(Y, s32), "auc_int8": ra(Y, s8),
                     "delta_pt": abs(ra(Y, s32) - ra(Y, s8)) * 100}
    # M4: FaceTrace val AUC fp32 vs int8 (ort sessions over the same eval set)
    try:
        import onnxruntime as ort
        from ml.train.facetrace import FaceDataset, prep, scores
        dev = "cuda"
        import timm
        from ml.train.facetrace import build
        m = build(pretrained=False)
        ck = torch.load("ml/runs/facetrace_v2/best.pt", map_location="cpu", weights_only=True)
        m.load_state_dict(ck["model"]); m.eval()
        ds = FaceDataset("data/manifests/video_real.parquet", "val", 600, True, 11, strength=0.35, augment=False)
        so = ort.SessionOptions(); so.intra_op_num_threads = 4
        a32 = ort.InferenceSession("models/facetrace.onnx", so, providers=["CPUExecutionProvider"])
        a8 = ort.InferenceSession("models/facetrace.int8.onnx", so, providers=["CPUExecutionProvider"])
        S32, S8, Y = [], [], []
        from torch.utils.data import DataLoader
        for xb, yb, _ in DataLoader(ds, batch_size=96, num_workers=4):
            x = prep(xb).numpy()
            S32.append(a32.run(None, {"x": x})[0].reshape(-1))
            S8.append(a8.run(None, {"x": x})[0].reshape(-1))
            Y.append(yb.numpy())
        Y = np.concatenate(Y); S32 = np.concatenate(S32); S8 = np.concatenate(S8)
        from sklearn.metrics import roc_auc_score as ra
        res["M4"] = {"n": int(len(Y)), "auc_fp32": float(ra(Y, S32)), "auc_int8": float(ra(Y, S8)),
                     "delta_pt": float(abs(ra(Y, S32) - ra(Y, S8)) * 100)}
    except Exception as e:
        res["M4"] = {"error": f"{e.__class__.__name__}: {e}"}
    print("INT8 AUC gate:", json.dumps(res, indent=2, default=float))
    Path("ml/reports/int8_gate.json").write_text(json.dumps(res, indent=2, default=float))
    keep = {k: ("int8" if v.get("delta_pt", 99) <= 0.5 else "fp32") for k, v in res.items() if "error" not in v}
    print("keep:", keep)
    return res, keep
