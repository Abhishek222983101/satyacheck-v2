"""P10.2 — latency budget on CPU only (blueprint A1: p95 < 8 s for a 10 s clip).
CUDA hidden, ORT CPU provider, 4 intra-op threads, sequential (one request at a time).
Writes p50/p95 into ml/reports/benchmark.json (latency_cpu) and prints a per-stage table.
Run: .venv/bin/python -m tests.latency_budget  (or pytest tests/latency_budget.py -s)
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # GPU hidden — this is the deployment claim
os.environ.setdefault("SATYA_RUNTIME", "ort")

N = int(os.environ.get("LAT_N", "50"))


def measure() -> dict:
    import glob
    from satya_core.onnx_models import OnnxModels
    from satya_core.verdict import Fuser
    from satya_core.pipeline import analyze_path

    paths = sorted(glob.glob("data/attacks/wav2lip/*/gen_0.mp4"))[: N // 2] + \
        sorted(glob.glob("data/attacks/wav2lip/*/gen_5.mp4"))[: N - N // 2]
    if len(paths) < 10:
        raise SystemExit("need >=10 clips under data/attacks/wav2lip")
    m, f = OnnxModels(), Fuser("models/nyaya.pt")
    m.m1(); m.m2(); m.m2_thr(); m.m4()  # warm: session creation + first allocations
    analyze_path(m, paths[0], fuser=f)  # warm inference
    totals, pre = [], []
    for p in paths:
        t = time.time()
        out = analyze_path(m, p, fuser=f)
        totals.append((time.time() - t) * 1000)
        pre.append(sum(v for k, v in out["timings_ms"].items() if k.startswith("pre_")))
        print(f"{Path(p).parent.name[:40]:40s} {totals[-1]:7.0f} ms (pre {pre[-1]:5.0f})", flush=True)
    p50 = statistics.median(totals)
    p95 = sorted(totals)[int(0.95 * len(totals)) - 1]
    res = {"n": len(totals), "p50_ms": round(p50), "p95_ms": round(p95),
           "mean_ms": round(statistics.mean(totals)), "preprocess_p50_ms": round(statistics.median(pre)),
           "gpu": "hidden", "threads": 4, "note": "10 s window cap; wav2lip attack clips as load"}
    print(json.dumps(res, indent=1))
    return res


def main():
    res = measure()
    gate = res["p95_ms"] < 8000
    print("GATE p95 < 8000 ms:", "PASS" if gate else "FAIL")
    b = Path("ml/reports/benchmark.json")
    if b.exists():
        bench = json.loads(b.read_text())
        bench["latency_cpu"] = res
        bench["latency_gate_pass"] = gate
        b.write_text(json.dumps(bench, indent=2, default=str))


if __name__ == "__main__":
    main()
