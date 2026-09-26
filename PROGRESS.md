# PROGRESS — SatyaCheck v2 (one line per sub-phase; ✅ done · 🟡 partial · ⬜ todo · ❌ blocked)
Blueprint: SATYACHECK_BLUEPRINT.md · Env: `. .venv/bin/activate` · Secrets: `source ~/.config/satya/wandb.env` (never commit)

## P0 Foundations
- ✅ P0.1 toolchain — py3.11 venv, torch 2.11+cu128 (sm_120 OK on RTX 5050), ORT 1.30, mediapipe 1.0.1, av 18.1
- ✅ P0.2 skeleton + Makefile + PROGRESS
- ✅ P0.3 weights (light set) — face_landmarker, XLS-R-300M, W2V2-AASIST, whisper-small, silero (6.1 GB total in ~/.cache/satya/weights, SHA256SUMS written). AV-HuBERT-L (3.9 GB) + AVH-Align NOT yet fetched — decide at P2.3
- ✅ P0.3b MediaPipe 1.0.1 smoke test: 478 landmarks @ ~16 ms/frame. FINDING: full-frame detector misses small faces (ai-anchor 0/60); 3×3 upscaled-tile retry → 10/10. Baked into blueprint §4.2.4
- ✅ W&B key valid (user abhisheksoni1445, entity abhisheksoni1445-mistralfluence); stored in ~/.config/satya/wandb.env (chmod 600), NOT in repo
- 🟡 P0.4 datasets — open slices downloading (logs/datasets.log): IndicTTS-DF 6 shards, IndicSynth hi+mr 3 shards each, LipFD val+ckpt, LipSyncTIMIT. Slow network (~1 MB/s). Gated sets (FakeAVCeleb/AV1M/HAV-DF) optional, not needed
- ✅ P0.5 contract — contracts/analysis.schema.json + fixture + 4 tests green (pydantic/zod codegen: TODO with P8/P9)
- ⬜ P0.6 recordings (human)
## P1 Data engine
- ✅ P1.2 ForwardChain emulator (priors) — ml/forwardchain/emulate.py; 5 gens in 2.8 s; deterministic per seed; ffmpeg 8 quirk: -r incompatible with vfr (handled). Replace priors via whatsapp_2026.json after P1.3
- ⬜ P1.1 manifests · ⬜ P1.3 WhatsApp calibration (optional/human) · ⬜ P1.4 ladders · ⬜ P1.5 splice factory · ⬜ P1.6 MuseTalk
  ## P2 Preprocess ⬜  ## P3 M2 SpliceLoc ⬜  ## P4 M1 SyncFormer ⬜
## P5 M4 FaceTrace + M3 ForwardAge ⬜  ## P6 Nyaya ⬜  ## P7 Eval ⬜  ## P8 Server ⬜  ## P9 Frontend ⬜  ## P10 Ship ⬜
