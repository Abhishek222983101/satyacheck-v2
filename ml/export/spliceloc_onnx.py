"""ONNX export of SpliceLoc (fixed 10 s window + explicit valid-length input) and CPU parity/latency check.
Usage: python -m ml.export.spliceloc_onnx --ckpt ml/runs/spliceloc_v3/best.pt [--int8]"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from ml.train.spliceloc import SpliceLoc

WIN = 160000  # 10 s @ 16 kHz (PS scope)

class Infer(nn.Module):
    """wav (B,160000) float32, n_valid (B,) int64 -> boundary logits (B,T), frame logits (B,T,2), utt logits (B,2)"""
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, wav, n_valid):
        T = (wav.shape[1] - 400) // 320 + 1
        valid = (torch.arange(T, device=wav.device)[None] < ((n_valid[:, None] - 400) // 320 + 1)).float()
        o = self.m(wav, n_valid, valid=valid)
        return o["boundary"], o["frame"], o["utt"]

def load(ckpt):
    m = SpliceLoc(); ck = torch.load(ckpt, map_location="cpu", weights_only=True); m.load_state_dict(ck["model"]); m.eval(); return Infer(m).eval(), ck

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", default="models/spliceloc.onnx"); ap.add_argument("--int8", action="store_true")
    a = ap.parse_args(); torch.set_num_threads(3)
    net, ck = load(a.ckpt); Path(a.out).parent.mkdir(exist_ok=True)
    wav = torch.randn(1, WIN) * 0.1; nv = torch.tensor([WIN - 12000])
    with torch.no_grad(): ref = [t.numpy() for t in net(wav, nv)]
    t0 = time.time()
    torch.onnx.export(net, (wav, nv), a.out, input_names=["wav", "n_valid"], output_names=["boundary", "frame", "utt"], opset_version=18, dynamo=False,
                      dynamic_axes={"wav": {0: "B"}, "n_valid": {0: "B"}, "boundary": {0: "B"}, "frame": {0: "B"}, "utt": {0: "B"}})
    print(f"exported in {time.time()-t0:.0f}s -> {a.out} ({Path(a.out).stat().st_size/1e6:.0f} MB)")
    import onnxruntime as ort
    def run(path, label):
        so = ort.SessionOptions(); so.intra_op_num_threads = 4
        s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"]); inp = {"wav": wav.numpy(), "n_valid": nv.numpy()}
        out = s.run(None, inp)
        ts = []
        for _ in range(3):
            t = time.time(); s.run(None, inp); ts.append(time.time() - t)
        d = [float(np.abs(o - r).max()) for o, r in zip(out, ref)]
        print(f"{label}: max|diff| boundary {d[0]:.4f} frame {d[1]:.4f} utt {d[2]:.4f} | 10 s clip CPU(4 threads): {min(ts):.2f}s")
        return out
    run(a.out, "fp32")
    if a.int8:
        # onnxsim first: folds the wav2vec pos_conv weight Mul into an initializer,
        # which quantize_dynamic needs (raises "Expected ... to be an initializer" without it)
        import onnx
        from onnxsim import simplify
        sim = a.out.replace(".onnx", ".sim.onnx")
        m, ok = simplify(onnx.load(a.out))
        assert ok, "onnxsim reported not-ok"
        onnx.save(m, sim)
        from onnxruntime.quantization import quantize_dynamic, QuantType
        q = a.out.replace(".onnx", ".int8.onnx")
        quantize_dynamic(sim, q, weight_type=QuantType.QInt8, per_channel=True)
        print(f"int8 -> {q} ({Path(q).stat().st_size/1e6:.0f} MB)"); run(q, "int8")

if __name__ == "__main__":
    main()
