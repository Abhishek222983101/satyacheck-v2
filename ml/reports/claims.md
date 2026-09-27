# Claims we can say — claim guard review (§4.11)

Evidence: `ml/reports/benchmark.json` (regenerate: `python -m ml.eval.benchmark`), git SHA in that file.
Every sentence below is backed by a measured number from the current harness. Anything not
listed here must NOT appear on a slide, README, or answer to a judge.

## Allowed (with their evidence)

| We say | Evidence | Caveat to volunteer if asked |
|---|---|---|
| "Nyaya fused AUC **0.845** on the held-out in-domain test (n=1,728, speaker-disjoint)" | nyaya.json → test.auc | fusion of 3 branches + gen-aware calibration |
| "Fusion beats every single branch (best single branch **0.772**, mean-of-branches **0.720**)" | ablations.json → baselines | every branch's removal hurts: no-sync 0.715, no-face 0.796, no-splice 0.807 |
| "Conformal coverage **0.951** at α=0.05 (target 0.95)" | nyaya.json → test.coverage | Gen4-5 bucket 0.928, slightly under target |
| "ECE **0.027** after temperature scaling (T=0.71)" | nyaya.json → test.ece | — |
| "On unseen-generator Wav2Lip fakes, fakes are called REAL in **0–3.1%** of cases across generations" | heldout_wav2lip.json → by_gen.*.fake_called_REAL | the model abstains heavily instead (~67–71% SUSPECT) — say this too |
| "FaceTrace AUC **0.956** on held-out speakers; falls to **0.937** only at extreme re-encoding" | PROGRESS P5.1 (val eval) | FaceTrace FPR on real climbs with generations (0.40→0.88) — disclosed weakness |
| "Pixel-only baseline (what everyone else does) drops **0.893 → 0.765** clean→extreme re-encoding; ours drops **0.971 → 0.937**" | pixel_only/metrics.json vs P5.1 | baseline trained Gen-0 only, same architecture/size |
| "SpliceLoc AUC **0.887**, F1@80 ms **0.62**, stable Gen0→Gen5" | spliceloc.json → mixed_all | measured on our T-SPL |
| "Spliced clips from the held-out Wav2Lip attack score AUC **0.86** (full) / **0.85** (partial) with FaceTrace" | heldout_wav2lip.json → by_kind | fusion on this set: 0.77/0.85 |
| "SyncFormer val AUC **0.887**, localisation AUC **0.80** on partial fakes" | PROGRESS P4.5 (W&B M1 run) | cost-volume offset within ±1 frame on 63% clean / 42% gen5 |
| "ForwardAge estimates emulated forward generation with MAE **0.80**" | forwardage.json | **emulated forwards only** — real-WhatsApp T-WA validation pending; top features are emulator parameters (honest caveat required) |

## Explicitly NOT allowed (until their gate passes)

- "Robust to 5th-generation WhatsApp forwards" — needs Gen-5 AUC within 5 pts of Gen-0 on T-LS and T-SPL **and** T-WA agreement. Measured Gen-0→Gen-5 in-domain: 0.871→0.793 (−7.8 pts) → **say the measured drop instead**.
- "Works on Hindi/Marathi" — T-LANG per-language table not yet built (splice by_lang exists; fused per-language does not).
- "Generalises to unseen generators" as an unqualified claim — Wav2Lip result exists but abstention is ~70%; allowed only with the abstention rate attached.
- "Real WhatsApp forwards validated" — T-WA empty until real phones forward clips (P1.3, human).
- Any "outperforms" against other papers' numbers — only against our own baselines (pixel-only, SyncNet zero-shot, mean-of-branches) on our test sets.
- "<10 s p95 on a mid-range laptop" — needs P10.2 latency run with GPU hidden (CUDA_VISIBLE_DEVICES="", 4 threads).

## Open measurements required before deploy claims

1. P7.1 `run_eval.py` end-to-end through ONNX (post P8.1) — regenerate benchmark from ONNX hashes.
2. P10.2 latency p95 on CPU-only runtime.
3. T-WA real-WhatsApp sanity check (needs phones; optional).
