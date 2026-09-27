# PROGRESS — SatyaCheck v2 (one line per sub-phase; ✅ done · 🟡 partial · ⬜ todo · ❌ blocked)
Blueprint: SATYACHECK_BLUEPRINT.md · Env: `. .venv/bin/activate` · Secrets: `source ~/.config/satya/wandb.env` (never commit)

## P0 Foundations
- ✅ P0.1 toolchain — py3.11 venv, torch 2.11+cu128 (sm_120 OK on RTX 5050), ORT 1.30, mediapipe 1.0.1, av 18.1
- ✅ P0.2 skeleton + Makefile + PROGRESS
- ✅ P0.3 weights (light set) — face_landmarker, XLS-R-300M, W2V2-AASIST, whisper-small, silero (6.1 GB total in ~/.cache/satya/weights, SHA256SUMS written). AV-HuBERT-L (3.9 GB) + AVH-Align NOT yet fetched — decide at P2.3
- ✅ P0.3b MediaPipe 1.0.1 smoke test: 478 landmarks @ ~16 ms/frame. FINDING: full-frame detector misses small faces (ai-anchor 0/60); 3×3 upscaled-tile retry → 10/10. Baked into blueprint §4.2.4
- ✅ W&B key valid (user abhisheksoni1445, entity abhisheksoni1445-mistralfluence); stored in ~/.config/satya/wandb.env (chmod 600), NOT in repo
- ✅ P0.4a audio sets — IndicTTS-DF (as/bn/brx), IndicSynth hi/mr (ALL synthetic), FLEURS hi/mr (real): 25.0 h after balancing
- 🟡 P0.4b video sets — LipFD val (3.7 GB) + LipSyncTIMIT via gdown, resumable, running (logs/video_sets.log)
- ✅ P0.5 contract — contracts/analysis.schema.json + fixture + 4 tests green (pydantic/zod codegen: TODO with P8/P9)
- ⬜ P0.6 recordings (human)
## P1 Data engine
- ✅ P1.2 ForwardChain emulator (priors) — ml/forwardchain/emulate.py; 5 gens in 2.8 s; deterministic per seed; ffmpeg 8 quirk: -r incompatible with vfr (handled). Replace priors via whatsapp_2026.json after P1.3
- ✅ P1.1 (audio half) manifests + group-disjoint splits + language balance (ml/data/audio_manifest.py, splits.py). Video half waits for video sets
- ✅ P1.5 splice factory (8 tests) + Whisper picker w/ hallucination guard + pause fallback
- ✅ P1.4 (audio) — superseded: training splices THEN applies in-memory AAC chain (ml/forwardchain/aac_mem.py, delay-compensated)
- ⬜ P1.3 WhatsApp calibration (human, optional) · ⬜ P1.4 video ladders · ⬜ P1.6 MuseTalk
  ## P2 Preprocess
- ✅ P2.1 satya_core/preprocess.py (tile-fallback tracking, 88x88 ROI, kinematics, mel, VAD) — verified visually on 3 samples
- ✅ P2.2 satya_core/features_codec.py (75-d; Benford dev rises 0.108→0.328 over 5 gens)
- ⬜ P2.3 teacher features (AV-HuBERT) DEFERRED by decision (fast path) · ⬜ P2.4 cache (needs video data)
## P3 M2 SpliceLoc
- ✅ P3.1 dataset (on-the-fly splice→AAC→crop; exact labels) · ✅ P3.2 model (117M total, 3.2M heads, XLS-R 8 layers, BAM-style heads)
- 🟡 P3.3 training v2 running: ml/runs/spliceloc_v2, W&B group M2-SpliceLoc, 9 epochs x 3000 (log: logs/spliceloc_v2.log)
- ⬜ P3.4 eval: `python -m ml.eval.eval_spliceloc --ckpt ml/runs/spliceloc_v2/best.pt` (script written, untested)
  ## P4 M1 SyncFormer ⬜
## P5 M4 FaceTrace + M3 ForwardAge ⬜  ## P6 Nyaya ⬜  ## P7 Eval ⬜  ## P8 Server ⬜  ## P9 Frontend ⬜  ## P10 Ship ⬜

## P4-P6 (M1 SyncFormer, M4 FaceTrace, M3 ForwardAge, M5 Nyaya) — status
- ✅ P4.0 real-speaker data: TalkVid-Bench (500) + 1,199 clips fetched via yt-dlp from TalkVid's metadata (900 speakers, 332 Hindi); 1,126 cached (819 train / 104 val / 107 cal / 96 test clips, speaker-disjoint). Ladder Gen1-5 + ROI/mel/kin per clip in data/cache_v.
- ✅ P4.1 dataset (real-only + synthetic shifts/swaps/partials, teacher targets), P4.2 model (7.1M params: 3D-stem+ResNet visual, conv-attn audio, 3 cross layers, kinematics, **cost-volume offset head**), P4.3 trained (runs A-D; final = ml/runs/syncformer_best.pt, epoch 15)
- ✅ P4.4 teacher: SyncNet (MIT) features on CLEAN clips for all 1,126 clips (median conf 6.7) -> compression-robust distillation
- ✅ P4.5 M1 val: AUC 0.887 (Gen0 0.92 -> Gen5 0.87-0.88); offset from cost volume within ±1 frame on 63% (Gen0) / 42% (Gen5); partial-fake localisation AUC 0.80
- ✅ P5.1 FaceTrace (EfficientNet-B0, self-blended pseudo-fakes, real-only): val AUC 0.956 (clean 0.971 / extreme compression 0.937 / lips-only 0.928)
- ✅ P5.2 ForwardAge (LightGBM ordinal, 75 codec features): EMULATED forwards only — MAE 0.80 gen, within-1 77%, forwarded-vs-original AUC 1.0; top features are resolution/bitrate/profile (emulator parameters) => NOT a real-WhatsApp measurement
- 🟡 P6.1 Nyaya samples: ml/train/run_nyaya_data.sh (val/cal/test, 6 conditions x Gen0-5) running; P6.2 fit+conformal: ml/train/nyaya.py (tested on synthetic data: 95.0% coverage at alpha 0.05)
- 🟡 Held-out generator: 88 Wav2Lip dubs of TEST clips forwarded Gen0-5 (data/attacks/wav2lip); end-to-end eval ml/eval/eval_heldout.py (+ SyncNet zero-shot baseline) after Nyaya fit
- Known issues / next: SpliceLoc ONNX export fails on pad-with-shape op (ml/export/spliceloc_onnx.py; fix at P8.1); preprocess latency 4-7 s/10 s clip dominates (P8); real WhatsApp forwards (T-WA) still needed to validate the emulator + ForwardAge
- Ops lessons: box has 15 GB RAM — cap concurrent workers (OOM killed runs twice); PyAV motion-vector export leaks ~35 MB/call (disabled, USE_MV=False); never `pkill -f` a pattern that appears in your own command line (use pid files + process groups)
