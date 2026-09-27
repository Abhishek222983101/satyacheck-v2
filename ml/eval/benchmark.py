"""P7.4 — assemble ml/reports/benchmark.json from the current measured reports.

Single source for the claim guard (§4.11): every sentence on a slide must come from
this file. Regenerate after any eval:  python -m ml.eval.benchmark
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

R = Path("ml/reports")


def _load(name: str):
    p = R / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _ckpt_sha(path: str) -> str | None:
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else None


def build() -> dict:
    ny, sp, fa = _load("nyaya"), _load("spliceloc"), _load("forwardage")
    ho, ab = _load("heldout_wav2lip"), _load("ablations")
    pixel = json.loads((Path("ml/runs/pixel_only") / "metrics.json").read_text()) if (Path("ml/runs/pixel_only") / "metrics.json").exists() else None
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()[:12]
    bench = {
        "generated_by": "python -m ml.eval.benchmark",
        "git_sha": git,
        "models": {k: _ckpt_sha(v) for k, v in {
            "syncformer": "ml/runs/syncformer_best.pt",
            "spliceloc": "ml/runs/spliceloc_v2/best.pt",
            "facetrace": "ml/runs/facetrace_v2/best.pt",
            "nyaya": "models/nyaya.pt",
            "pixel_only": "ml/runs/pixel_only/best.pt"}.items()},
        "in_domain": {
            "M1_syncformer": {"auc_val": 0.887, "note": "PROGRESS P4.5 measured (T-LS val)"},
            "M2_spliceloc": {"test": sp.get("mixed_all") if sp else None, "per_gen": sp.get("mixed_by_gen") if sp else None},
            "M4_facetrace": {"note": "val AUC 0.956 clean/extreme 0.971/0.937 (PROGRESS P5.1)"},
            "M5_nyaya": {"test": ny.get("test") if ny else None, "ablations": ab.get("in_domain_ablations") if ab else None,
                         "baselines": ab.get("baselines_in_domain") if ab else None},
            "M3_forwardage": {"note": fa.get("note") if fa else None, "test_mae_by_lang": fa.get("test_mae_by_lang") if fa else None},
        },
        "unseen_generator": {"wav2lip": ho} if ho else None,
        "baselines": {"pixel_only_gen0": pixel} if pixel else None,
    }
    (R / "benchmark.json").write_text(json.dumps(bench, indent=2, default=str))
    print("wrote", R / "benchmark.json", "git", git)
    return bench


if __name__ == "__main__":
    build()
