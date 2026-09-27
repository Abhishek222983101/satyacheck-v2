"""Fusion -> verdict. Uses ml.train.nyaya's own tensors/logit/bucket/set functions so training and serving cannot drift.
Returns the `verdict` block of the AnalysisResult contract (contracts/analysis.schema.json)."""
from __future__ import annotations

import numpy as np
import torch

from ml.train.nyaya import BR, Nyaya, bucket_of, logit, predict_sets

BUCKET = {0: "gen_0_1", 1: "gen_2_3", 2: "gen_4_5"}


class Fuser:
    def __init__(self, path="models/nyaya.pt"):
        ck = torch.load(path, map_location="cpu", weights_only=True)
        self.net = Nyaya(); self.net.load_state_dict(ck["model"]); self.net.eval()
        self.T, self.q, self.alpha = float(ck["T"]), {int(k): v for k, v in ck["qhat"].items()}, float(ck["alpha"])

    def __call__(self, b, meta: dict) -> dict:
        """b: satya_core.pipeline.BranchOut."""
        m1, m2, m4, m3 = b.m1, b.m2, b.m4, b.m3
        ok = np.array([m1["status"] == "ok", m2["status"] == "ok", m2["status"] == "ok", m4["status"] == "ok"], np.float32)
        raw = [m1.get("clip_score"), m2.get("p_spliced"), m2.get("p_synthetic"), m4.get("clip_score")]
        S = np.array([[float(logit(v)) if (o and v is not None) else 0.0 for v, o in zip(raw, ok)]], np.float32)
        gen_hat = float(m3["gen_hat"])
        C = np.array([[gen_hat / 5.0, float(meta.get("face_presence") or 0), float(np.clip(meta.get("speech_ratio") or 0, 0, 1))]], np.float32)
        with torch.no_grad():
            z, g = self.net(torch.from_numpy(S), torch.from_numpy(ok[None]), torch.from_numpy(C))
        p = float(torch.sigmoid(z / self.T)[0]); gate = g[0].numpy()
        if ok.sum() == 0:                                                     # nothing could be checked: say so, never guess
            reason = "no_speech" if not meta.get("has_audio") or meta.get("no_speech") else ("no_face" if meta.get("no_face") else "too_short")
            return dict(label="SUSPECT", p_fake=0.5, integrity=50, conformal_set=["REAL", "FAKE"], alpha=self.alpha, bucket=BUCKET[int(bucket_of(gen_hat))],
                        reason=reason, gate=gate.tolist())
        s = predict_sets([p], [gen_hat], self.q)[0]
        label = "FAKE" if s == {"FAKE"} else "REAL" if s == {"REAL"} else "SUSPECT"
        reason = None
        if label == "SUSPECT":
            reason = "heavy_degradation" if gen_hat >= 4.5 else "uncertain"
            hi = [v > 0.8 for v, o in zip([m1.get("clip_score"), m2.get("p_spliced"), m4.get("clip_score")], [ok[0], ok[1], ok[3]]) if o and v is not None]
            if len(hi) >= 2 and any(hi) and not all(hi): reason = "branch_conflict"
        return dict(label=label, p_fake=p, integrity=int(round(100 * (1 - p))), conformal_set=sorted(s), alpha=self.alpha, bucket=BUCKET[int(bucket_of(gen_hat))],
                    reason=reason, gate=gate.tolist())
