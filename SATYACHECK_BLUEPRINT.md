# SATYACHECK v2 — SYSTEM ARCHITECTURE & BUILD BLUEPRINT

> **Document:** `SATYACHECK_BLUEPRINT.md` · **Version** 2.0 · **Date** 2026-09-27
> **Team** Paradigm Shift · **Event** MUSA CodeX 2026 · **Problem statement** CX0102 — *Fake Face, Real Riot* (AI & ML domain)
> **Status** Pre-build. Design frozen for P0. **Supersedes** the v1 prototype (`lib/analyze.ts` heuristics) and the v1 README's architecture section.
> **Frozen external facts:** live URL `https://fake-face-riot-paradigm-shift.vercel.app` (submitted, must not change) · repo `github.com/Abhishek222983101/paradigm-shift-codex-2026` · demo video `youtu.be/lT2kp7NVi-c`.

---

## HOW TO READ THIS DOCUMENT

| If you are… | Read | Skip |
|---|---|---|
| **A teammate preparing the pitch** | Part 1, Part 2, Part 8 (demo + script). ~25 min. | Parts 5–6, 9. |
| **The person (or AI agent) building it** | **Everything.** Part 4 (the ML system) and Part 9 (the phase plan) are the working documents. Part 10 tells you which Claude model to run for each sub-phase. | Nothing. |
| **A judge who asks "show me"** | Part 4.10 (evaluation protocol) and Part 4.11 (the claim guard). | — |

**Conventions used throughout:**

| Marker | Meaning |
|---|---|
| `[PS]` | Traces directly to the CX0102 problem statement. Not negotiable. |
| `[JUDGE:<criterion>]` | Traces to one of the eight published judging criteria. |
| `[DECISION]` | A design decision we made. Rationale always given. Can be revisited with reason. |
| `[RESEARCH]` | Grounded in a cited paper/repo (Part 3 and §12.6 list sources). |
| `[TARGET]` | A number we are *aiming* for. Never shown to judges as a result. |
| `[MEASURED]` | A number produced by our own evaluation harness (P7). **Only these go on slides.** |
| `[RISK]` | A known failure mode with a stated mitigation. |
| `[CUT]` | Sits below the cut line — build only if time remains (see §9.0). |
| 🟣 Opus / 🔵 Sonnet / 🟢 Haiku | Which Claude model to run that sub-phase with (Part 10). |

**The single rule this document enforces:** *every claim on the stage must be reproducible by `make eval` on this laptop.* A deepfake detector that bluffs about its own accuracy is exactly the kind of manipulated media the problem statement is about. The judges in an ML track can smell a bluff; they cannot argue with a live ROC curve.

---

# PART 1 — WHAT WE ARE BUILDING, IN PLAIN LANGUAGE

## 1.1 The problem, stated honestly `[PS]`

A doctored video of a local leader spreads on WhatsApp in a small town and triggers unrest before fact-checkers can react. The original clip is never found. So:

- there is **no reference** to compare against (no "find the original" shortcut),
- the clip is **6–10 seconds long** (very little signal),
- it is **vernacular** (Hindi, Marathi, Bhojpuri, Tamil… — English-trained ASR and lip-readers degrade),
- it has been **re-compressed 3–5 times** by forwarding (the twist) — every forward decodes, downscales, re-encodes video and audio, and wipes the pixel-level traces most detectors rely on,
- the fact-checker has a **mid-range laptop and <10 seconds**,
- and the two attacks named are **lip-sync mismatch** (new words put in a real mouth — Wav2Lip / MuseTalk / LatentSync class) and **audio splicing** (real words cut and re-joined: *"I do **not** support this decision"* → *"I do support this decision"*).

## 1.2 Where v1 stands — the honest audit

v1 (Round 2) is a Next.js page with ~660 lines of hand-tuned heuristics in the browser. It was the right prototype and it is the wrong finale product:

| v1 behaviour | Why it will not survive an ML-track judge |
|---|---|
| "Lip-sync" = frame-difference motion in 9 fixed image zones correlated with audio loudness | No face detection, no learned representation. Any moving object near the speaker scores as "mouth". |
| "Splice" = spectral-flux peaks with a MAD threshold | Hand-tuned; fires on expressive speech (hence the "dense-prosody guard" hack). |
| "Generation age" = 4 `if` statements (height ≤ 500, bitrate < 900…) | Not learned, not validated, and "Force Gen-5" literally sets `gen = 5`. |
| Fusion = fixed 45/35/20 weights | No calibration, so the 0–100 "integrity" number means nothing statistically. |
| All three demo clips are **mute** | The FAKE on the AI anchor comes from a *"mouth moves + no audio → −8"* rule — not from detecting anything. |
| "Force lip-sync fake" shifts the loudness curve and injects a splice marker | The detector reacting to numbers we injected is not a detection. |
| "Synthetic REAL/FAKE" returns hard-coded scores | A judge who reads the code sees a mock. |
| The Round-1 deck promised Whisper, wav2vec2, fine-tuning on FakeAVCeleb, an ONNX build, AUC on clean vs Gen-5, a WhatsApp generational-loss simulator | **None of it exists.** Round 1 is on record. |

**What v1 got right, and v2 keeps:** the three-question framing (does the mouth move *with* the voice? is the audio one continuous take? how many forwards old is it?), the abstain-instead-of-guess principle, the <10 s headline with visible per-stage timings, the evidence-board UX, and the frozen URL. v1's heuristics become **`baseline-v1`** — a row in our benchmark table and the offline fallback in the browser (§5.6).

## 1.3 What we are building — one sentence

**SatyaCheck v2 is a compression-robust, self-supervised audio-visual forensics system that learns what a *real* speaking face sounds and moves like, is trained *through* a measured WhatsApp forwarding chain, localises *where* in a 6–10 s clip the lip-sync or splice happened, and returns a calibrated, abstain-when-unsure verdict on a CPU in under 10 seconds.**

Every word in that sentence is a mechanism in Part 4:

| Phrase | Mechanism |
|---|---|
| *self-supervised … learns what real looks like* | SyncFormer is pre-trained on **real** talking heads only, with an audio-shift-prediction objective (WACV 2026) and distilled from AV-HuBERT / AVH-Align (trained on 45k real videos). Real-only training is what generalises to generators we have never seen (SAVe 2026, AVH-Align CVPR 2025). |
| *compression-robust … trained through a measured WhatsApp chain* | **ForwardChain** — our own WhatsApp generational-loss emulator, calibrated against clips we actually forwarded 5× between phones (SNVSE method, ≥30 clips per resolution). Every model trains on Gen-0…Gen-5 versions. Research shows real codec diversity beats fancy frequency architectures for compression robustness (+7.3 AUC points, arXiv 2608.28685). |
| *localises where* | Frame-level heads: SyncFormer outputs a per-40 ms sync-error curve; SpliceLoc outputs per-20 ms spoof + boundary probabilities (BAM-style boundary-aware attention). |
| *calibrated, abstain-when-unsure* | **Nyaya** fusion: temperature-scaled probabilities + **Mondrian split-conformal** abstention per compression bucket — a statistical guarantee on the false-FAKE rate among the clips we *do* label. |
| *CPU in under 10 seconds* | Teachers (≈1 B params total) stay offline; distilled students (~115 M params total) are exported to **ONNX INT8** and run in parallel threads. |

## 1.4 The five models, in one table

| | Model | Answers | Kind of ML | Deployed size |
|---|---|---|---|---|
| **M1** | **SyncFormer** | *Does the mouth move with this voice — and where does it stop?* | Self-supervised cross-modal transformer (3D-CNN lip encoder + audio encoder + cross-attention), shift-prediction pre-training, knowledge-distilled from AV-HuBERT-Large, fine-tuned on lip-sync fakes | ~12 M |
| **M2** | **SpliceLoc** | *Is this one continuous recording? Is the voice synthetic? Where is the cut?* | Layer-truncated multilingual wav2vec2 (XLS-R) front-end + boundary-aware frame attention (BAM) + utterance anti-spoof head (AASIST-distilled) | ~100 M |
| **M3** | **ForwardAge** | *How many times has this been forwarded, and on what platform?* | Gradient-boosted ordinal regressor on codec-domain features (QP stats, macroblock types, motion vectors, DCT first-digit statistics, AAC stats) — ground truth comes free from ForwardChain | ~2 MB |
| **M4** | **FaceTrace** | *Was this face generated or blended?* (covers fully synthetic "AI anchor" videos that have no audio at all) | EfficientNet-B0 trained with Self-Blended Images (real-only pseudo-fakes) + Gen-0…5 augmentation | ~5 M |
| **M5** | **Nyaya** | *Given all of that — and given how degraded the clip is — what can we honestly say?* | Compression-gated fusion MLP + temperature scaling + Mondrian conformal abstention | <0.1 M |

Plus two offline systems that are research contributions in their own right:

- **ForwardChain** — the WhatsApp generational-loss emulator (data engine + live demo).
- **RedTeam Forge** — our attack factory: MuseTalk lip-sync fakes on Indic speakers, cut-and-join splices, Indic TTS insertions. It makes our training fakes *and* powers the live "fake the judge" demo.

## 1.5 What it is not

| It is not | Why |
|---|---|
| A transcript-based detector | Whisper/lip-reading in Marathi or Bhojpuri at Gen-5 is unreliable. We are *language-agnostic by construction*: verdicts come from AV physics and acoustics. Whisper is used only to **annotate** the evidence card ("the word *nahin* sits at the splice"), never to decide. |
| A face-swap detector first | The PS names lip-sync and splicing. Face-swap/full synthesis is covered by M4, but M1/M2 are the heart. |
| A single black-box score | Every verdict ships with a timeline, a localisation, the generation estimate and the reason it abstained. |
| An LLM wrapper | No API calls at inference. Everything runs locally on CPU. |
| A claim of "99% accuracy" | We report our own measured numbers — including where we are weak (Part 4.11). |

## 1.6 What "done" means

Done is a live, unrehearsed run in front of a judge where all of the following happen on one laptop:

1. A judge picks any clip (ours, theirs, or **a clip of themselves we record on the spot**).
2. Within 10 s, SatyaCheck returns REAL / SUSPECT / FAKE, with per-branch timings on screen.
3. For a spliced clip, the waveform shows the cut **within ±80 ms** and Whisper names the words either side.
4. For a lip-sync fake, SyncScope shows the sync-error curve spiking over the dubbed words and plays the clip **re-aligned by the model's predicted offset**.
5. The ForwardChain Theatre forwards that same clip 5× live and the verdict holds, while the baseline column collapses.
6. The Benchmark Wall shows the ROC/AUC-per-generation numbers produced by `make eval` — the same command a judge could run.

---

# PART 2 — JUDGING MAP & ACCEPTANCE TESTS

## 2.1 The eight criteria, each tied to something a judge will see

Source: hackathon site "Judging Criteria — Mentors and judges evaluate every submission across eight dimensions" (no published weights). Finale: 27 Sep 2026, Grand Golden Banquet, Mumbai; top-45 teams "build live".

| Criterion | What earns it | The on-stage proof | Part |
|---|---|---|---|
| **Innovation** | Real-only self-supervised training; ForwardChain calibrated on *real* WhatsApp forwards; conformal abstention; attack-the-judge demo | RedTeam live splice of the judge's own sentence → caught | 4.2, 4.3, 4.8, 8.2 |
| **Problem Understanding** | The twist made first-class (Gen-age is a model output, not a caveat); vernacular-first by design; abstention for mute/no-face clips | Chain Theatre: Gen-0→Gen-5 verdict curve vs baseline | 1.1, 4.2, 8.2 |
| **Technical Implementation** | 4 trained models + teachers + distillation + ONNX INT8 + measured eval | Benchmark Wall + Model X-ray | 4, 7.5 |
| **Scalability** | Stateless CPU workers, ~₹0.02/clip `[ESTIMATE]`, browser tier, WhatsApp tipline design | Architecture slide + cost table | 5.5, 8.4 |
| **Practical Impact** | Built for a fact-checker's laptop; bilingual evidence dossier they can forward back into the same WhatsApp group | Dossier PDF/PNG in Hindi + English | 7.4, 8.2 |
| **Presentation** | 5-minute script with one "wow" per minute | Part 8.3 script | 8 |
| **User Experience** | 3-click flow; one-tap samples; record-yourself; honest empty/error/abstain states | Desk → Case → Dossier | 7 |
| **Teamwork** | Clear ownership per model/branch (maps to deck's 4 roles) | Team slide with who built which model | 8.3 |

## 2.2 Acceptance tests — the contract

Each row must be demonstrable live. Each has an automated test in `tests/` and/or `ml/eval/`.

| # | Requirement | Demonstration | Automated check |
|---|---|---|---|
| **A1** `[PS]` | Verdict in < 10 s on a mid-range laptop | Any 10 s clip, CPU-only, timings visible | `test_latency_budget` — p95 < 8.0 s over 50 clips with ONNX threads pinned to 4, GPU disabled |
| **A2** `[PS]` | Detects lip-sync mismatch | RedTeam MuseTalk fake of an Indic speaker → FAKE, sync curve spikes on dubbed span | `eval_sync` AUC on held-out lip-sync set (identity-disjoint) reported per generator |
| **A3** `[PS]` | Detects audio splicing, localised | Judge sentence with one word removed → cut marked within ±80 ms | `eval_splice` boundary F1@±80 ms, frame EER |
| **A4** `[PS]` | Works at 4th–5th generation | Chain Theatre Gen-0…5 | `eval_generations` — AUC per Gen; target drop Gen-0→Gen-5 ≤ 5 pts `[TARGET]` |
| **A5** `[PS]` | 6–10 s clips | All test clips cut to 6–10 s | eval harness samples 6–10 s windows only |
| **A6** | Honest abstention | Mute clip → SUSPECT with reason "no speech", not FAKE | `test_abstain_reasons`; conformal risk on accepted set ≤ α on calibration split |
| **A7** | Vernacular | Hindi/Marathi clips in the test set, results broken out by language | `eval_by_language` table |
| **A8** | Reproducible claims | `make eval` regenerates `ml/reports/benchmark.json` and every number on the Benchmark Wall | CI-style script; page reads the JSON, never hard-codes |

---

# PART 3 — RESEARCH GROUNDING

*What the state of the art says, what we adopt, what we reject, and why. Full links in §12.6.*

## 3.1 Audio-visual deepfake detection

| Work | Key idea | Reported | Our use |
|---|---|---|---|
| **AVFF** (CVPR 2024) | Stage-1 self-supervised AV correspondence (contrastive + autoencoding, complementary masking) on real videos; stage-2 supervised | 99.1% AUC FakeAVCeleb | Inspires our 2-stage recipe. Unofficial code `JoeLeelyf/OpenAVFF`. Too heavy to deploy. |
| **AVH-Align** (Smeu et al., CVPR 2025) | AV-HuBERT features + small alignment MLP trained with InfoNCE **on real videos only**; shows supervised datasets contain shortcuts (e.g. leading silence) | Pretrained on 45k real AV-Deepfake1M videos (`checkpoints/AVH-Align_AV1M.pt`) | **Teacher for M1.** Its real-only prior is exactly what an unseen Indic generator needs. |
| **SAVe** (2026) | Real-only: self-blended face/lip/lower-face pseudo-fakes + AV-HuBERT AVSync branch, average-logit fusion | AVSync alone AUC 0.94 on FakeAVCeleb; Wav2Lip 0.96 | Validates real-only + late fusion. Our M4 uses the same self-blend family. |
| **A-V shift prediction** (WACV 2026) | Pre-train by predicting the temporal shift applied to audio vs. video; frozen features then detect **and localise** | SOTA within-dataset, strong generalisation + localisation | **Pre-training objective for M1** — gives us a per-frame offset estimate for free (powers the "re-sync" demo). |
| **LipFD** (NeurIPS 2024) | Global AV temporal features + region-aware encoder; AVLips dataset (340k samples) | 95.3% avg acc; 90.2% on real WeChat calls; ~310M params | Baseline + AVLips data source. |
| **LIPINC-V2** (2025) | Vision temporal transformer on mouth RGB + delta frames; LipSyncTIMIT (5 generators, CRF 23/40) | SOTA on LipSyncTIMIT/FakeAV-LS/KoDF-LS | Baseline + LipSyncTIMIT data. Note: adding audio naively *hurt* them — why we use a learned sync space, not concatenation. |
| **Investigating SSL representations for AV deepfake detection** (CVPR 2026) | Audio-informed visual features (AV-HuBERT-V, BRAVEn-V) generalise best; real-only anomaly detection is robust to spurious cues | SOTA with simple heads | Justifies AV-HuBERT as the distillation target. |

## 3.2 Temporal localisation (partial manipulation)

| Work | Reported | Our use |
|---|---|---|
| **AV-Deepfake1M** (ACM MM 2024) — 1M+ videos, 2,068 subjects, word-level replace/insert/delete | SOTA localisers collapse: UMMAFormer AP@0.5 98.83 (LAV-DF) → 51.64 (AV1M) | Our threat model is exactly this: *one word changed*. We report localisation separately and honestly. |
| **BA-TFD+ / UMMAFormer** | Boundary-matching + temporal-anomaly attention | Design reference for frame heads. |

## 3.3 Audio anti-spoofing & splice localisation

| Work | Reported | Our use |
|---|---|---|
| **W2V2(XLS-R 300M)-AASIST** (Tak et al., Odyssey 2022) — `huggingface.co/SpeechAntiSpoofingBenchmarks/W2V2-AASIST` | SOTA ASVspoof 2021 LA/DF; 317.8 M params | **Teacher for M2 utterance head.** |
| **BAM — Boundary-aware Attention** (Interspeech 2024) — `github.com/media-sec-lab/BAM` | PartialSpoof EER 3.58%, F1 96.09% @160 ms; boundary EER 3.33% | **Architecture for M2 frame + boundary heads.** |
| **Waveform boundary detection** (ADD 2022) | wav2vec2 + ResNet-1D + Transformer + BiLSTM, EER 6.58% | Alternative head, ablation row. |
| **Layer-wise SSL analysis** (NAACL Findings 2025) | Lower/middle SSL layers carry most anti-spoofing signal | Justifies **truncating XLS-R to its first 8 layers** → CPU-feasible. |
| **AntiDeepfake XLS-R** (NII Yamagishi lab) — trained on 56k h real + 18k h fake, 100+ languages | Strong zero-shot | Optional second teacher. |

## 3.4 Compression robustness — the twist

| Work | Finding | Our use |
|---|---|---|
| **"Data Diversity, Not Frequency Invariance"** (arXiv 2608.28685, Aug 2026) | Plain EfficientNet-B0 on multi-quality data beat a DCT/FFT-phase compression-invariant architecture by 3.66 AUC at CRF 40; **real CRF variants beat synthetic JPEG augmentation by 7.3 points** | **The core [DECISION] behind ForwardChain**: spend effort on realistic codec chains, not exotic frequency branches. |
| **SNVSE** (ACM MM 2025) — `github.com/truebees-ai/social_emulator` | Emulate a platform by estimating CRF + resize from ≥30 uploaded/downloaded pairs per resolution; detectors fine-tuned on emulated ≈ on real shared videos | **Method for calibrating ForwardChain against real WhatsApp.** |
| **Robustness study** (Frontiers AI, 2026) | Avg accuracy 94.7% clean → 67.8% distorted; temporal+spatial models degrade least | The collapse we demo with the baseline column. |
| **PLADA** (2025) | Block effects mimic forgery traces; must not be confused with them | Why ForwardAge feeds *into* fusion as context (the model must know how degraded the clip is). |
| **WhatsApp forensics** (UC Denver theses 2020) | Android re-encodes HD+ uploads to H.264 with the long side capped (≈640 px then), CFR→VFR, AAC re-encode; file-structure signatures identify WhatsApp | Priors for ForwardChain; **we re-measure on 2026 builds** because apps change. |

## 3.5 Indian-language data

| Dataset | What | Access |
|---|---|---|
| **InDeepFake** (Pattern Recognition Letters 2025) | AV deepfakes, **7 Indian languages**, 7 generators, 4.6k videos | `github.com/arnabdasphd/InDeepFake` — open |
| **HAV-DF** (2024) | Hindi AV deepfake: 200 real + 308 fake (face-swap, lip-sync, RVC voice clone) | Paper arXiv 2411.15457 — verify access |
| **IndicTTS Deepfake Challenge** | 31,102 labelled clips, **16 Indian languages**, TTS vs real | `huggingface.co/datasets/SherryT997/IndicTTS-Deepfake-Challenge-Data` — open |
| **IndicSynth** (ACL 2025) | 4,000+ h synthetic speech, 12 Indian languages (XTTS-v2, VITS, FreeVC) over IndicSUPERB reals | `huggingface.co/datasets/vdivyasharma/IndicSynth` — open |
| **MLADDC** (NeurIPS 2024 D&B) | 20 languages (6 Indian) incl. "half-truth" partial fakes | Verify access |

## 3.6 Generators for our own fakes

| Tool | Fit on an 8 GB RTX 5050 | Use |
|---|---|---|
| **MuseTalk 1.5** (Tencent, open weights + training code) | Runs on 4 GB (≈5 min per 8 s clip on a 3050 Ti); faster here | **Primary lip-sync generator** for RedTeam Forge + training fakes. |
| **LatentSync 1.5** (ByteDance) | Needs 8 GB for inference — borderline | Secondary / held-out generator (test-only → proves generalisation). |
| **Wav2Lip** | Trivial | Legacy generator, test-only. |
| **XTTS-v2 / IndicSynth voices** | CPU/GPU fine | Synthetic word insertion for splice training. |

## 3.7 Calibration & abstention

Split conformal prediction and selective classification give **distribution-free guarantees** on the error rate among the predictions a model chooses to make (Angelopoulos & Bates; *Calibrated Selective Classification*, 2022; *SCoRE*, 2026). Deepfake detectors are known to be over-confident on unseen generators (*Towards reliable deepfake detection from an uncertainty-calibration perspective*, 2025). → **Nyaya** (§4.8).

## 3.8 What we deliberately reject

| Rejected | Why |
|---|---|
| Training a 300 M-param end-to-end AV transformer | Not trainable in our window on 8 GB, not runnable in 10 s on a CPU. We *distil* instead. |
| Transcript-mismatch detection (Whisper vs lip-reading, "Lost in Translation" CVPRW 2024) as the verdict | Lip-reading models are English-trained; at Gen-5 in Marathi they are noise. Kept as *annotation*. |
| An LLM "explains" the verdict | Hallucination risk in a misinformation tool is unacceptable. Explanations are generated from measured evidence with templates. |
| Frequency-domain magic branches | §3.4: data diversity wins; we spend the time on ForwardChain. |
| Uploading clips to a cloud GPU | Privacy + venue Wi-Fi + the PS's laptop constraint. |


---

# PART 4 — THE ML SYSTEM

## 4.1 Five design principles (the pitch in five lines)

1. **Learn real, not fake.** Generators change monthly; real speech physics does not. Every branch has a real-only pre-training stage; fakes are used only to fine-tune heads. `[RESEARCH]` AVH-Align, SAVe, CVPR 2026 SSL study.
2. **Train through the forward chain.** Every sample is seen at Gen-0…Gen-5 of a *measured* WhatsApp pipeline. The model is never surprised by degradation. `[RESEARCH]` arXiv 2608.28685, SNVSE.
3. **Localise, don't just classify.** A fact-checker needs "the cut is at 3.4 s, between these two words", not "73% fake". Frame-level heads everywhere.
4. **Know how degraded you are.** ForwardAge's generation estimate is an *input* to fusion; confidence is calibrated per quality bucket.
5. **Abstain with a reason.** Mute track, no face, conformal set = {REAL, FAKE} → SUSPECT with a machine-readable reason, never a coin flip.

## 4.2 The data engine

### 4.2.1 Sources and roles

| Pool | Source | Role | Size target |
|---|---|---|---|
| **R-AV** real talking heads | AVLips real split, LipSyncTIMIT real (VidTIMIT), InDeepFake real, HAV-DF real, **own recordings** (team + friends, Hindi/Marathi/English, phone cameras, 6–12 s, consented) | M1 self-supervised pre-training, M4 self-blend, calibration | 3–6 k clips `[TARGET]` |
| **F-LS** lip-sync fakes | AVLips fake, LipSyncTIMIT fakes (Diff2Lip, VideoRetalking, Wav2Lip, Wav2Lip-GAN, IP_LAP), InDeepFake lip-sync subset, **RedTeam Forge MuseTalk fakes on R-AV Indic speakers** | M1 fine-tune + eval | 3–5 k |
| **F-HELD** held-out generator | LatentSync fakes (made by us) + one LipSyncTIMIT generator excluded from training (IP_LAP) | **Test only** — the generalisation claim | 300–500 |
| **R-A** real speech | IndicSUPERB via IndicSynth bona fide, IndicTTS-challenge real, R-AV audio tracks | M2 | 20–40 k utterances |
| **F-A** synthetic speech | IndicTTS-challenge TTS, IndicSynth (XTTS-v2 / VITS / FreeVC), PartialSpoof (English, has segment labels) | M2 utterance + frame heads | 20–40 k |
| **S-A** splices (generated) | RedTeam splice factory (§4.2.3) over R-A | M2 boundary head | 30 k+ (free, on the fly) |
| **G** generation ladders | ForwardChain over everything above | All models + M3 labels | ×6 per clip |

`[RISK]` Gated datasets (FakeAVCeleb, AV-Deepfake1M, LRS3, VoxCeleb2) need licence forms that may not return in time. **Mitigation:** the plan does not depend on them. AVH-Align's weights already encode 45k real AV1M videos; open pools above are sufficient. Request them anyway in P0 (§9, P0.4).

`[DECISION]` **Identity-disjoint splits.** Train/val/cal/test are split by *speaker identity*, never by clip. Leaking identities inflates AUC by 10+ points and is the first thing a sharp judge asks about.

Split: 70 / 10 / 10 (conformal calibration) / 10 (test) by identity, stratified by language and generator.

### 4.2.2 ForwardChain — the WhatsApp generational-loss emulator `[PS twist]`

**Why it is a contribution:** published emulators (SNVSE) cover Facebook/YouTube/BlueSky — the SNVSE paper explicitly lists WhatsApp as future work. We measure WhatsApp ourselves, and we model **chains** (Gen-n), not a single share.

**Calibration protocol (P1.3, ~1.5 h of phone time, parallelisable by the team):**

1. Take 36 source clips: 12 × 1080p, 12 × 720p, 12 × 480p; mixed content (talking head, crowd, low light, screen recording).
2. Forward each clip phone A → phone B → phone A … **5 hops**, alternating Android and iPhone where possible, default quality ("Standard") and once with "HD".
3. Export every hop via WhatsApp Web (unmodified downloads — see §3.4 forensics: download method does not alter hashes).
4. `ml/forwardchain/measure.py` runs `ffprobe` on every hop: resolution, fps mode, video bitrate, profile/level, GOP, B-frames, audio codec/rate/bitrate/channels, container box order.
5. Fit per-hop parameters with the SNVSE procedure: for each (source, hop) pair, re-encode the source over CRF 18…40 with libx264 and pick the CRF whose bitrate is just under the WhatsApp output → an empirical distribution `p(CRF | input_res, hop)`; scale rule `p(long_side_out | long_side_in)`; audio `p(aac_bitrate)`.
6. Save `ml/forwardchain/whatsapp_2026.json`. **This file is a slide.** ("We forwarded 36 clips 180 times. Here is what WhatsApp actually does in 2026.")

**Emulator (`ml/forwardchain/emulate.py`):** for Gen-n, apply the hop transform n times, drawing parameters per hop from the fitted distributions:

```
hop(x):
  v = scale(x, long_side ~ p(long_side_out | in))        # lanczos/bicubic randomly (encoder variance)
  v = fps_jitter(v, cfr→vfr with prob p_vfr)
  v = libx264(v, crf ~ p(crf|res,hop), preset ∈ {veryfast,faster}, profile ∈ {baseline,main}, g ∈ [30,250])
  a = aac(x.audio, br ~ p(aac_br), sr ∈ {44100,48000}, ch = 1 w.p. p_mono)
  with prob 0.08: screen_rerecord(v)   # re-captured on another phone: slight crop, moiré, gamma
  with prob 0.05: sticker/caption overlay burn-in
  return mux(v, a, faststart)
```

Fallback priors (used until our calibration lands, from the literature and ffprobe of known WhatsApp files): long side 640–848, CRF 28–34, AAC-LC 48–96 kbps.

**Outputs:** `(clip_id, gen, params_json, sha256)` rows → `data/manifests/gens.parquet`. M3's labels are the true `gen` values — no manual labelling.

**Throughput:** `ffmpeg` with `-threads 2`, 12 parallel workers on 12 cores ≈ 150–250 short clips/min `[TARGET]`. 5k clips × 5 gens ≈ 2 h wall clock — run in the background (🟢 Haiku can monitor).

### 4.2.3 RedTeam Forge — the attack factory

| Attack | How | Labels produced | Also a live demo? |
|---|---|---|---|
| **Cut-and-join splice** (the riot attack) | Force-align words with Whisper (word timestamps) → delete / swap / duplicate 1–3 words → join with 0–15 ms crossfade, optional room-tone fill, loudness match | Frame labels + boundary timestamps | **Yes — instant** |
| **Cross-take splice** | Join segments from two recordings of the same speaker (different mic/room) | Frame + boundary | Yes |
| **TTS word insertion** | Clone the speaker with XTTS-v2 (Hindi supported) or use an IndicSynth voice; insert 1–3 synthetic words | Frame (spoof) + boundary | Pre-made |
| **Lip-sync dub** | MuseTalk 1.5 driven by (a) the speaker's own re-ordered audio, (b) another speaker, (c) TTS | Video-level + span labels (dubbed segment) | Pre-made (≈1–3 min/clip on GPU) |
| **Partial lip-sync** | Dub only a 1.5–3 s window; keep the rest real (AV-Deepfake1M style) | Span labels | Pre-made |
| **Held-out** | LatentSync 1.5 dubs — never used in training | Test only | Yes (shows generalisation) |

`[DECISION]` Splices are generated **on the fly in the dataloader** (different cut every epoch) — effectively infinite, label-perfect training data for M2.

`[RISK]` "Leading/trailing silence" shortcut (documented by AVH-Align): fakes in public sets often have tell-tale silence padding. **Mitigation:** trim leading/trailing silence from *all* clips, randomise crop start, loudness-normalise to −23 LUFS.

### 4.2.4 Preprocessing contract (identical in training and inference)

Implemented once in `satya_core/preprocess.py`, imported by both training and the server — the single most important anti-bug rule.

| Step | Spec |
|---|---|
| Decode | PyAV; video → 25 fps CFR (frame-time resample, not interpolation); audio → 16 kHz mono float32 |
| Window | Assess the first 10.0 s (PS scope); clips < 2 s → abstain `too_short` |
| Face | MediaPipe Face Landmarker (478 pts + 52 blendshapes), `VIDEO` mode, every frame; track largest face with IoU continuity; **if the full frame yields no face (wide shots, small faces — the AI-anchor sample is one: 0/60 frames), retry on 3×3 overlapping 50% tiles upscaled to 512 px with confidences 0.3 (validated: 10/10 sampled frames)**; `face_presence < 0.5` for >40% of frames → M1/M4 abstain `no_face` |
| Mouth ROI | Similarity-align to a mean face using landmarks (eyes + nose tip), crop 96×96 around lip centre, grayscale, then center-crop 88×88 (AV-HuBERT convention) |
| Face crop | 224×224 aligned face for M4, 16 frames sampled uniformly |
| Lip kinematics | Per-frame: inner-lip aperture (landmarks 13/14), mouth width (61/291), `jawOpen`, `mouthClose`, `mouthFunnel`, `mouthPucker` blendshapes → 8-dim vector (cheap, compression-proof auxiliary input to M1) |
| Audio features | 80-bin log-mel, 25 ms window / 10 ms hop (for M1 audio encoder); raw waveform (for M2) |
| Speech activity | Silero-VAD (ONNX, 2 MB); speech ratio < 15% → M1/M2 abstain `no_speech` |
| Codec features | `ffprobe` stream info + PyAV side data (motion vectors with `flags2=+export_mvs`) + per-frame QP via `-debug qp` fallback; DCT stats from decoded I-frames (§4.6) |

## 4.3 M1 — SyncFormer (lip-sync mismatch)

### 4.3.1 Architecture

```
 mouth ROI 88×88×T (25 fps)          log-mel 80×4T (100 fps)        lip kinematics 8×T
          │                                   │                              │
  3D-conv stem (5×7×7, 64ch)          2× Conv1d stride 2 → 100→25 fps       Linear 8→64
  ResNet-18 2D trunk per frame         + 4-layer Conformer-lite (d=256)      │
  (shared across frames) → 256-d                  │                          │
          │                                   a_t (256)                      k_t
          └──── v_t (256) ⊕ k_t ────┐           │
                                    ▼           ▼
                   Cross-modal Transformer: 4 layers, d=256, 4 heads,
                   interleaved self-attn per stream + cross-attn A↔V,
                   local attention window ±32 frames (1.28 s)
                                    │
        ┌──────────────┬────────────┴─────────────┬───────────────────────┐
        ▼              ▼                          ▼                       ▼
  Head S: sync     Head Δ: shift            Head F: frame fake       Head D: distil
  InfoNCE over     classification over       prob per frame          regress AV-HuBERT-L
  ±15-frame        {-15..+15} frames         (fine-tune only)        layer-12 V & A feats
  neighbourhood    (per 16-frame segment)                             (teacher, offline)
```

- Params ≈ 11–13 M. Input 10 s = 250 frames.
- `[DECISION]` Grayscale 88×88 mouth ROI (not full face): compression-robust, identity-light, matches the teacher's view.
- `[DECISION]` Kinematic landmark stream: mouth aperture survives Gen-5 better than texture, and fakes often get the *shape* right and the *timing* slightly wrong — kinematics make timing cheap to learn.

### 4.3.2 Training recipe (three stages)

**Stage A — cache teacher features (offline, GPU).** For every R-AV clip at Gen-0 and a random Gen-k: run **AV-HuBERT-Large** (`self_large_vox_433h.pt`, fp16, ~4 GB VRAM) → per-frame visual + audio features (1024-d, layer 12); run the **AVH-Align** MLP → per-frame alignment score. Store as `float16` `.npy` shards. (≈0.3 s per clip on the 5050 → 5k clips ≈ 30 min.)

**Stage B — self-supervised pre-training on R-AV only (all generations).**
- Loss `L = L_nce + λ_shift·L_shift + λ_kd·L_kd`, with λ_shift = 0.5, λ_kd = 1.0 → 0.3 (cosine decay).
- `L_nce`: InfoNCE with the true-offset audio frame as positive, ±15-frame neighbours as hard negatives (AVH-Align formulation), temperature 0.07.
- `L_shift`: shift audio by s ~ U{−15…+15} frames (±600 ms) per 16-frame segment; cross-entropy over 31 classes (WACV 2026). Also 20% of batches with **swapped audio from another speaker** → class "no-sync" (32nd class).
- `L_kd`: cosine distance between projected student features and cached AV-HuBERT features — the student inherits a 325 M model's speech representation.
- Augmentation: ForwardChain gen ∈ {0…5} (uniform), random 6–10 s crops, mouth ROI jitter ±4 px, brightness/contrast, 10% frame drops (simulating VFR).
- Optim: AdamW lr 3e-4, wd 0.05, cosine, 3 k warm-up steps, bf16 AMP, batch 24 × 250 frames (fits 8 GB with gradient checkpointing on the trunk), 25–40 epochs `[TARGET]` ≈ 2–3 h.

**Stage C — supervised fine-tune (frozen stem, train transformer top-2 + heads).**
- Data: F-LS + R-AV, balanced 1:1, generations mixed.
- Head F loss: frame BCE (dubbed span = 1) + video-level BCE on max-pooled frames; focal γ = 2 for class imbalance of partial fakes.
- Keep `L_shift` at λ 0.2 to prevent forgetting (it powers the re-sync demo).
- 5–8 epochs, lr 1e-4 ≈ 40 min.

### 4.3.3 Inference outputs

| Output | Definition | Shown as |
|---|---|---|
| `sync_curve[t]` | 1 − softmax prob of the zero-offset class, smoothed over 5 frames | SyncScope red curve |
| `offset_ms` | argmax of Head Δ aggregated over segments × 40 ms | "Best alignment: +320 ms" + **re-sync playback** |
| `p_fake_video` | Head F max-pool → calibrated in Nyaya | Branch score |
| `spans` | Frames with Head F > τ merged (min 200 ms) | Red bands on timeline |
| `av_heatmap[offset, t]` | Head Δ probability matrix (31 × segments) | The "sync fingerprint" heatmap |

## 4.4 M2 — SpliceLoc (audio splicing + synthetic voice)

### 4.4.1 Architecture

```
 raw waveform 16 kHz (≤10 s)
      │
 XLS-R-300M, **layers 0–8 only** (CNN feature encoder + 8 of 24 transformer layers)
      │  (weighted sum of layers 4–8, learnable softmax weights)        → 50 fps × 1024
 Linear 1024→256 + attentive pooling to 20 ms (50 fps kept) and 160 ms views
      │
 ┌────┴──────────────────────────────────────────────┐
 Boundary Enhancement (BAM):                          │
   intra-frame 1D-ResNet + inter-frame attention      │
      │ → boundary logits b_t (per 20 ms)             │
 Boundary-aware Frame Attention ×2 (mask cross-segment│
   message passing using predicted boundaries)        │
      │ → frame spoof logits f_t                      │
      └──────────── concat ───────────────┐           │
                                          ▼           ▼
                           Utterance head: attentive stats pool → MLP → p_synthetic
                           + distillation target: W2V2-AASIST teacher logit
```

- Params: ≈ 95 M in the truncated XLS-R + ≈ 3 M heads. `[RESEARCH]` lower/middle SSL layers carry most spoofing evidence → truncation costs little and saves ~65% compute.
- `[DECISION]` Multilingual XLS-R (128 languages incl. Hindi, Marathi, Tamil, Bengali) instead of English wav2vec2-base.

### 4.4.2 Training recipe

- **Data mix per batch:** 40% on-the-fly splices of real Indic speech (S-A), 25% TTS / partial-TTS (IndicTTS challenge, IndicSynth, PartialSpoof), 35% untouched real. All passed through ForwardChain audio at random gen (AAC re-encode chain) + RawBoost (linear/non-linear convolutive noise) at p = 0.5.
- **Losses:** `L = CE(frame) + 0.5·BCE(boundary, ±2-frame label dilation) + 0.5·BCE(utterance) + 0.5·KD(utterance, AASIST teacher, T = 2)`.
- **Schedule:** freeze XLS-R for 1 epoch (heads warm-up, lr 1e-3), then unfreeze layers 5–8 at lr 1e-5 with LLRD 0.8; bf16; batch 16 × 6 s; gradient checkpointing; 6–10 epochs `[TARGET]` ≈ 2–3 h.
- **Teacher:** W2V2-AASIST LA checkpoint run once over F-A + R-A → cached logits.

### 4.4.3 Inference outputs

`frame_spoof[t]` (20 ms), `boundaries[] = {t, strength}` via peak-picking on b_t with NMS (min gap 300 ms), `p_synthetic`, `p_spliced = max_t b_t` (calibrated). Plus, **asynchronously after the verdict** (not in the 10 s budget): Whisper-small word timestamps → the dossier names the words on either side of each boundary.

## 4.5 M4 — FaceTrace (face synthesis / blending artifacts)

- **Why it exists:** the "AI anchor" sample is fully generated and *mute*. v1 caught it with a rule. v2 catches it because the face itself carries generation/blending artifacts — and says so.
- **Backbone:** EfficientNet-B0 (ImageNet init), 224×224 aligned face crops, 16 frames/clip, clip score = mean of top-4 frame logits.
- **Training:** Self-Blended Images (SBI, Shiohara & Yamasaki CVPR 2022 — `github.com/mapooon/SelfBlendedImages`) generated on the fly from **R-AV faces** (real-only pseudo-fakes; region masks: full face, lower face, lips — the SAVe region set), plus the fully-synthetic subsets available in AVLips / InDeepFake. Every sample passes through ForwardChain video at a random gen **before** the crop (so the model sees the true codec, not a resized PNG).
- **Recipe:** AdamW lr 1e-4, bf16, batch 64, 10–15 epochs ≈ 1 h. Grad-CAM on the last conv block → heat overlay.
- **Outputs:** `p_face_fake`, per-frame scores, Grad-CAM maps for 4 frames (PNG artifacts), `face_quality` (landmark confidence × crop sharpness) for fusion gating.
- `[RISK]` Pixel artifacts fade by Gen-4/5 (§3.4). **Mitigation:** that is expected and *reported*; Nyaya down-weights M4 when ForwardAge says Gen ≥ 4 (learned, not hand-set).

## 4.6 M3 — ForwardAge (generation & platform forensics)

- **Features (≈180-d, all cheap):**
  - Container: box order signature (WhatsApp mux has a recognisable `moov`/`mdat` layout), encoder tag, fps mode (CFR/VFR).
  - Stream: resolution, long side, bitrate/pixel, profile/level, GOP length, B-frame ratio.
  - Bitstream: per-frame QP mean/std/histogram (10 bins), I/P/B macroblock-type ratios, skip-block ratio, motion-vector magnitude histogram (8 bins).
  - Double-compression cues: first-digit (Benford) distribution of quantised DCT coefficients on I-frames (9-d per 3 AC frequency bands), 8×8 blockiness ratio, Laplacian variance (v1's features, now just inputs).
  - Audio: AAC bitrate, spectral cutoff frequency (each AAC re-encode lowers the bandwidth — a strong generation clock), mono/stereo.
- **Model:** LightGBM, ordinal formulation (5 binary "gen > k" classifiers → expected gen) + a platform head {WhatsApp, Telegram-like, original, other}. Trained on ForwardChain ladders (labels are free). ≈ 10 min training.
- **Outputs:** `gen_hat` (0–5, continuous) + distribution, `platform` probs, SHAP top-3 features → the dossier text ("audio bandwidth cut to 7.2 kHz — typical of 3+ AAC re-encodes").
- `[DECISION]` M3 does not decide fake/real. It tells fusion **how much to trust the other branches**, and it answers the twist directly.

## 4.7 Parallel runtime & latency budget

Branches run concurrently in a `ThreadPoolExecutor` (ONNX Runtime releases the GIL), each ORT session with `intra_op_num_threads = 2`.

| Stage | Work | Budget (4-core laptop CPU, INT8) `[TARGET]` |
|---|---|---|
| Decode + resample | PyAV, 10 s | 0.4 s |
| Face landmarks | MediaPipe on every 2nd frame (125) + interpolate | 0.8 s |
| ROI / crops / VAD / codec feats | numpy + Silero ONNX + ffprobe | 0.4 s |
| M1 SyncFormer | 250 frames | 0.9 s ┐ |
| M2 SpliceLoc | 10 s audio | 1.4 s ├ parallel → ~1.5 s |
| M4 FaceTrace | 16 crops | 0.5 s │ |
| M3 ForwardAge | LightGBM | 0.05 s ┘ |
| Nyaya + artifacts | fusion, conformal, PNG heatmaps | 0.3 s |
| **Total** | | **≈ 3.9 s** (A1 requires p95 < 8 s) |

`[RISK]` A judge's laptop / venue machine is slower. **Mitigation:** each stage streams its result over SSE as soon as it finishes; budget guard skips M4's extra frames and uses 8 crops if elapsed > 6 s; every run shows the real timings.

## 4.8 M5 — Nyaya (fusion, calibration, conformal abstention)

### 4.8.1 Inputs

```
z = [ logit(p_sync_fake), logit(p_spliced), logit(p_synthetic), logit(p_face_fake),
      gen_hat, speech_ratio, face_presence, face_quality, snr_estimate,
      availability mask (m1, m2, m4) ]              # 13-d
```

### 4.8.2 Model

- **Compression-gated fusion:** `g = softmax(W_g · [gen_hat, face_quality, speech_ratio, mask])` produces per-branch weights; `logit = Σ g_i · a_i · branch_logit_i + MLP(z)` (2 layers, 32 hidden). Unavailable branches are masked to zero weight (so a mute clip is scored only by M4 + M3 context).
- Trained on the **val** split (never on train outputs → avoids over-confident stacking).
- **Temperature scaling** on val → `p_fake`. Report ECE before/after.

### 4.8.3 Mondrian split-conformal abstention

- Calibration split (identity-disjoint, 10%), partitioned into buckets by `gen_hat` ∈ {0–1, 2–3, 4–5}.
- Non-conformity score for label y: `s = 1 − p(y)`. Per bucket, q̂ = the ⌈(n+1)(1−α)⌉/n quantile, α = 0.05 `[DECISION]`.
- Prediction set `C(x) = { y : 1 − p(y) ≤ q̂_bucket }`:
  - `{FAKE}` → **FAKE**
  - `{REAL}` → **REAL**
  - `{REAL, FAKE}` or `∅` → **SUSPECT** (reason `uncertain`)
- **Guarantee (marginal, per bucket):** P(true label ∈ C(x)) ≥ 95%. **Consequence to say on stage:** "When SatyaCheck commits to FAKE or REAL, the true label is inside its answer at least 95% of the time — separately for fresh clips and for 5th-forward clips. When it cannot guarantee that, it says SUSPECT."
- Reason codes (the UI shows the first that applies): `no_speech`, `no_face`, `too_short`, `uncertain`, `branch_conflict` (M1 says real, M2 says spliced), `heavy_degradation` (gen_hat ≥ 4.5 and set size 2).
- The **integrity score** shown to users = `round(100 · (1 − p_fake))`, now a calibrated probability, not a weighted average.

## 4.9 Explainability layer

| Artifact | Source | Visual |
|---|---|---|
| Sync-error curve + dubbed spans | M1 heads S/F | SyncScope timeline (red) over speech energy (blue) |
| AV offset heatmap | M1 head Δ | 31-row heatmap, the diagonal "lock" of real speech vs. smeared fakes |
| **Re-sync playback** | M1 `offset_ms` | Video plays with audio shifted by the predicted offset — lips snap into place for delay-type fakes |
| Splice boundaries + frame spoof | M2 | Waveform with boundary pins + spoof heat under the wave |
| Word context | Whisper-small (async) | "…मैं इस फैसले का ✂ समर्थन करता हूँ" with the cut marked |
| Face heat | M4 Grad-CAM | 4-frame strip + 3D mesh heat (Part 7) |
| Generation reasoning | M3 SHAP | "Gen 4.3 ± 0.6 — audio bandwidth 7.2 kHz, QP mean 34, WhatsApp mux signature" |
| Decision trace | Nyaya | Gate weights per branch + conformal set + reason code |

`[DECISION]` All natural-language explanations are **templated from measured fields**. No LLM text generation in the verdict path.

## 4.10 Evaluation protocol (`ml/eval/`, `make eval`)

**Test sets (identity-disjoint):**

| Set | Contents | Purpose |
|---|---|---|
| T-LS | F-LS test + matching reals, Gen-0…5 | Lip-sync (A2, A4) |
| T-SPL | Held-out Indic speakers, generated splices + TTS inserts, Gen-0…5 audio | Splice (A3, A4) |
| T-HELD | LatentSync + IP_LAP fakes (never trained on) | Generalisation to unseen generators |
| T-WA | ~40 clips truly forwarded 5× on real phones (from P1.3) | **Real-WhatsApp sanity check** of the emulator claim |
| T-LANG | By language: hi / mr / en / other | Vernacular (A7) |
| T-MUTE | Mute / no-face / music clips | Abstention correctness (A6) |

**Metrics:** AUC, AP, EER, accuracy@τ (τ from val), FPR@95%TPR; **AUC per generation** (the twist curve); frame-level EER + **boundary F1@±80 ms** (M2); span IoU AP@0.5 (M1); ECE before/after temperature; conformal **coverage** and **abstention rate** per bucket; latency p50/p95 (ORT, 4 threads, GPU hidden via `CUDA_VISIBLE_DEVICES=""`).

**Baselines (same test sets, same windows):** `baseline-v1` (our own prototype heuristics, run headless via Playwright or a Python port), **AVH-Align zero-shot**, **W2V2-AASIST zero-shot**, **LIPINC-V2 pretrained** `[CUT]`, **LipFD pretrained** `[CUT]`, and a **pixel-only EfficientNet trained on Gen-0 only** (the "what everyone else does" column that collapses in the Chain Theatre).

**Ablations (one row each):** − ForwardChain augmentation (Gen-0 only) · − shift pre-training · − distillation · − kinematic stream · − gated fusion (mean of branches) · − conformal (plain threshold).

**Output:** `ml/reports/benchmark.json` (+ PNG plots: ROC per set, AUC-vs-generation, reliability diagram, latency histogram). The web app's Benchmark Wall renders this JSON directly.

## 4.11 The claim guard

| We say | Only if |
|---|---|
| "X% AUC on …" | It is in `benchmark.json` from the current model hashes. |
| "robust to 5th-generation WhatsApp forwards" | Gen-5 AUC on T-LS and T-SPL is within 5 points of Gen-0 **and** T-WA agrees. Otherwise we say the measured drop. |
| "works on Hindi/Marathi" | T-LANG shows it, per language, with n. |
| "generalises to unseen generators" | T-HELD AUC is reported next to in-domain. |
| "outperforms" | Against the named baselines above, on our test sets. Never against papers' self-reported numbers on other datasets. |
| "< 10 s on a mid-range laptop" | p95 from `test_latency_budget` with GPU hidden and 4 threads. |

Weak numbers are presented as findings ("face artifacts fade after Gen-4 — which is exactly why we fuse with sync and splice cues"). Judges reward honesty they can verify; they punish bluffs they can catch.

## 4.12 Model export & packaging

- PyTorch → `torch.onnx.export(dynamo=True)`, opset 18, dynamic axes on time; `onnxsim`; **dynamic INT8 quantisation** (`onnxruntime.quantization.quantize_dynamic`, per-channel, MatMul/Conv) for M1/M2/M4; fp32 kept for heads if INT8 drifts > 0.5 AUC.
- Parity test: max |logit diff| PyTorch vs ORT < 1e-2 (fp32) / AUC delta < 0.5 pt (INT8) on 200 val clips.
- `models/registry.json`: `{name, version, sha256, onnx_path, input_spec, calib: {T, qhat_by_bucket}, trained_on, metrics_ref}`. The server refuses to start if a hash mismatches (no silent model swaps before the demo).
- Distribution: models (~150 MB INT8 total) go to a Hugging Face model repo (`paradigm-shift/satyacheck-v2`) and are pulled by `make models`. Not committed to git.


---

# PART 5 — SYSTEM ARCHITECTURE

## 5.1 The one-page architecture

```
                                   ┌──────────────────────────────────────────────────────────┐
  Judge / fact-checker browser     │  apps: Next.js 16 (App Router) · React 19 · TS · R3F     │
  (frozen URL on Vercel, or        │  Desk · Case board · Chain Theatre · RedTeam · Lab       │
   localhost at the venue)         │  SSE client · MediaPipe preview · ORT-web fallback tier  │
                                   └───────────────┬───────────────────────▲──────────────────┘
                                     HTTPS (REST)  │                       │ SSE stream (stage events,
                                     multipart     │                       │ branch results, artifacts)
                                                   ▼                       │
          Cloudflare Tunnel (https://api.<name>.trycloudflare.com)  ── or ──  localhost:8000 at venue
                                                   │
┌──────────────────────────────────────────────────▼──────────────────────────────────────────────────┐
│  satya-server  (FastAPI · Uvicorn · Python 3.11)                                                     │
│                                                                                                      │
│  /v1/analyses ─► Job runner (asyncio + ThreadPool) ─► satya_core.pipeline                             │
│                                                        │                                             │
│      preprocess (PyAV, MediaPipe, Silero-VAD, ffprobe) │                                             │
│            ├──────────────┬──────────────┬─────────────┴──┐                                          │
│            ▼              ▼              ▼                ▼                                          │
│      M1 SyncFormer   M2 SpliceLoc   M4 FaceTrace     M3 ForwardAge        (ONNX Runtime, INT8, CPU)  │
│            └──────────────┴──────┬───────┴────────────────┘                                          │
│                                  ▼                                                                   │
│                        M5 Nyaya fusion + conformal ─► explain (templated) ─► artifacts (PNG/JSON)    │
│                                                                                                      │
│  /v1/chain  ─► ForwardChain emulator (ffmpeg) ─► re-score each Gen via same pipeline                  │
│  /v1/redteam ─► splice factory (instant) · MuseTalk worker (GPU, async)                              │
│  /v1/benchmarks, /v1/atlas ─► reads ml/reports/*.json                                                │
│  async post-verdict: Whisper-small word timestamps ─► dossier                                        │
│                                                                                                      │
│  SQLite (cases, analyses, branch_results, artifacts, chains, attacks)   ·   ./var/artifacts/…         │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘

  Offline (same repo, `ml/`): data engine · ForwardChain · RedTeam Forge · teachers (AV-HuBERT-L,
  AVH-Align, W2V2-AASIST) · training (PyTorch 2.7+ cu128, Lightning, Hydra) · eval · ONNX export · registry
```

## 5.2 Key architecture decisions

| # | Decision | Why |
|---|---|---|
| D1 | **Keep the Next.js app at the repo root**; add `server/`, `ml/`, `satya_core/`, `contracts/` beside it | Vercel's project root is the repo root and the URL is frozen. Moving the app to `apps/web` means changing Vercel settings on finale day. Not worth it. |
| D2 | **Inference on a Python server, not in the browser** | AV-HuBERT-distilled students, MediaPipe, PyAV, codec side-data and ffmpeg are Python-native. 10 s CPU budget is met server-side. The browser gets a fallback tier (D5). |
| D3 | **One shared `satya_core` package** used by training *and* serving | Train/serve skew in preprocessing is the #1 silent killer of ML demos. |
| D4 | **SSE streaming** of per-stage results | The judge sees branches finish one by one. Waiting on a spinner looks like nothing is happening. |
| D5 | **Browser fallback tier**: ORT-web (WASM) running M4-INT8 + the v1 heuristics, used when the API is unreachable | The frozen URL must never show a dead page, even if the laptop tunnel drops. Also: "privacy mode — nothing leaves the device". |
| D6 | **SQLite, not Postgres** | Single laptop, one writer, zero ops. Schema is Postgres-compatible for the scale story. |
| D7 | **Public reach via Cloudflare Tunnel** from the laptop; **Hugging Face Space (CPU)** as the always-on mirror `[CUT]` | Vercel serverless cannot run 150 MB of models + ffmpeg within limits; Vercel page is HTTPS, so a plain `http://` laptop API would be blocked as mixed content — the tunnel gives HTTPS. |
| D8 | **No auth** for the demo; per-case unguessable UUIDv7 IDs; uploads deleted after 24 h | Hackathon scope; privacy by retention limit. |

## 5.3 Request lifecycle (one analysis)

```
browser                         server                                   pipeline
  │ POST /v1/analyses (file)      │                                          │
  │──────────────────────────────►│ validate (≤100 MB, video/*, ≤5 min)      │
  │◄──── 202 {id, events_url} ────│ insert analyses(status=queued)            │
  │ GET events_url (SSE)          │ submit job ─────────────────────────────►│ preprocess
  │◄── event: stage {decode}──────│◄──────────────────────── progress cb ────│
  │◄── event: stage {landmarks} ──│                                          │ fan-out M1 M2 M3 M4
  │◄── event: branch {M3 …} ──────│◄─────────────────── M3 done (fastest) ───│
  │◄── event: branch {M4 …} ──────│                                          │
  │◄── event: branch {M1 …} ──────│                                          │
  │◄── event: branch {M2 …} ──────│                                          │ Nyaya + explain
  │◄── event: verdict {…} ────────│ update analyses(status=done), artifacts  │
  │◄── event: annotation {words}──│ (async Whisper, +2–4 s)                  │
  │◄── event: end ────────────────│                                          │
```

## 5.4 Deployment topology (finale day)

| Environment | Frontend | API | Use |
|---|---|---|---|
| **Stage laptop (primary)** | `next start` on `localhost:3000` | `uvicorn` on `localhost:8000` | The live demo. Zero network dependency. |
| **Frozen URL** | Vercel (existing project) with `NEXT_PUBLIC_API_BASE` = tunnel URL | Cloudflare Tunnel → stage laptop | Judges browsing on their phones; the submission link. |
| **Mirror** `[CUT]` | same Vercel build | HF Space (CPU basic, Docker) | If the laptop sleeps. |
| **Fallback** | Vercel | none | ORT-web tier + v1 heuristics, banner "offline mode". |

`[RISK]` `trycloudflare.com` quick-tunnel URLs change on restart. **Mitigation:** the frontend reads `/api-base.json` from Vercel at runtime (`public/api-base.json`), updated with one `vercel --prod` redeploy (≈40 s) — or use a named tunnel on a free Cloudflare account for a stable hostname.

## 5.5 Scalability story (for the judge who asks)

- Stateless inference workers (ORT, CPU) behind a queue (Redis + Arq) → horizontal scale; ≈4 CPU-seconds per 10 s clip → one 4-vCPU box ≈ 700 clips/hour → **~₹0.02 per clip** at cloud on-demand prices `[ESTIMATE]`.
- Dedup by perceptual hash (pHash of 8 key frames + audio chromaprint): viral clips are scanned once, answered forever — the property that matters for WhatsApp virality.
- Distribution channel (future scope, designed): WhatsApp Business tipline — forward a clip to a number, receive the dossier image back; newsroom API returning `AnalysisResult` JSON; browser extension.
- Edge: the browser tier (D5) proves on-device feasibility; M1/M4 INT8 are < 20 MB each.

## 5.6 Technology choices, with the reason for each

| Layer | Choice | Reason |
|---|---|---|
| Web framework | **Next.js 16.3 + React 19.2** (existing) | Frozen Vercel project; App Router; RSC for static pages (Lab, Method) |
| Styling | **Tailwind CSS v4** (existing) + CSS custom-property tokens | Already in the repo |
| 3D | **three + @react-three/fiber + @react-three/drei** | Face-mesh heat and embedding atlas (Part 7) |
| Motion | **motion** (Framer Motion successor) | Stage transitions, verdict seal |
| Charts | **d3-scale / d3-shape** + custom SVG (no chart kit) | Full control of forensic plots; small bundle |
| Audio viz | **wavesurfer.js v7** + regions plugin | Waveform + splice pins + playback |
| Client face preview | **@mediapipe/tasks-vision** (Face Landmarker, GPU delegate) | Live landmark overlay on the clip and the webcam |
| Browser inference | **onnxruntime-web** (WASM, WebGPU when available) | Fallback tier (D5) |
| State / data | **zustand** + **@tanstack/react-query**; native **EventSource** for SSE | Minimal, predictable |
| Validation / types | **zod** schemas generated from `contracts/analysis.schema.json` | One contract for Python + TS |
| i18n | **next-intl** (en, hi, mr) | Vernacular UI + dossier |
| Server | **FastAPI + Uvicorn**, **sse-starlette**, **SQLModel** (SQLite) | Async SSE, typed models |
| Media | **ffmpeg 6+/7**, **PyAV**, **ffprobe** | Decode, emulator, codec side-data |
| Vision | **mediapipe** (Python), **opencv-python-headless** | Landmarks, alignment |
| Audio | **torchaudio / soundfile**, **silero-vad (ONNX)**, **faster-whisper** (CTranslate2, small, int8) | VAD; word timestamps |
| Inference | **onnxruntime 1.2x** (CPU EP; CUDA EP optional) | INT8 on CPU, GIL-free threads |
| Training | **PyTorch ≥ 2.7 with CUDA 12.8 wheels** (required for RTX 50-series / Blackwell sm_120), **Lightning 2.x**, **Hydra** configs, **timm** (EfficientNet), **transformers** (XLS-R), **fairseq-free AV-HuBERT loader** (vendored), **LightGBM**, **scikit-learn** | Standard, well-supported |
| Tracking | **Weights & Biases** (`wandb`, project `satyacheck`, key via `~/.netrc` — never in the repo) | Every run's config, curves, confusion matrices, ROC and artifact hashes; the W&B dashboard is a judge-facing exhibit (§7.3.5) and the Lab page embeds run links |
| Data | **pyarrow/parquet** manifests, **DVC** `[CUT]` | Reproducible splits |
| Packaging | **uv** (Python 3.11 venv — system Python is 3.14, which torch/mediapipe wheels do not target) | Fast, lockfile |
| Reports | **WeasyPrint** (PDF) + **Pillow** (shareable PNG card) | Dossier |
| Tunnel | **cloudflared** | HTTPS reach to the laptop |

## 5.7 Repository structure

```
paradigm-shift-codex-2026/
├── app/                         # Next.js (stays at root — D1)
│   ├── layout.tsx               # fonts, i18n provider, theme tokens
│   ├── page.tsx                 # Desk (landing = live instrument)
│   ├── case/[id]/page.tsx       # Evidence board
│   ├── chain/page.tsx           # ForwardChain Theatre
│   ├── redteam/page.tsx         # Attack lab
│   ├── lab/page.tsx             # Model X-ray + Benchmark Wall
│   ├── method/page.tsx          # How it works (static, RSC)
│   └── globals.css              # tokens
├── components/                  # UI components (Part 7.6)
│   ├── desk/ case/ chain/ redteam/ lab/ three/ ui/
├── lib/
│   ├── api.ts  sse.ts  store.ts  format.ts
│   ├── contracts.ts             # generated zod types
│   ├── analyze.ts               # v1 heuristics = baseline-v1 + offline tier (kept, untouched)
│   └── offline/ort-facetrace.ts # browser fallback tier
├── public/
│   ├── samples/                 # v1 samples + new speaking samples (§8.1)
│   ├── models/facetrace-int8.onnx
│   └── api-base.json
├── contracts/
│   └── analysis.schema.json     # single source of truth (JSON Schema 2020-12)
├── satya_core/                  # shared Python package (D3)
│   ├── preprocess.py  features_codec.py  vad.py
│   ├── models/{syncformer,spliceloc,facetrace,forwardage,nyaya}.py   # ORT wrappers
│   ├── pipeline.py  explain.py  conformal.py  registry.py
├── server/
│   ├── main.py  routes/{analyses,chain,redteam,benchmarks,health}.py
│   ├── jobs.py  db.py  storage.py  dossier.py
│   └── workers/musetalk_worker.py
├── ml/
│   ├── forwardchain/{measure.py,fit.py,emulate.py,whatsapp_2026.json}
│   ├── redteam/{splice.py,tts_insert.py,musetalk_batch.py,latentsync_batch.py}
│   ├── data/{download/*.sh,manifests.py,splits.py,dataset_*.py}
│   ├── teachers/{avhubert_extract.py,avh_align.py,aasist_logits.py}
│   ├── train/{syncformer.py,spliceloc.py,facetrace.py,forwardage.py,nyaya.py}
│   ├── conf/…yaml               # Hydra
│   ├── eval/{run_eval.py,baselines/*.py,plots.py}
│   ├── export/{to_onnx.py,quantize.py,parity.py}
│   └── reports/benchmark.json + *.png
├── models/registry.json         # hashes; ONNX files pulled by `make models`
├── tests/{test_contract.py,test_latency_budget.py,test_abstain_reasons.py,e2e.spec.ts}
├── Makefile  pyproject.toml  uv.lock  package.json
└── SATYACHECK_BLUEPRINT.md (this file)  README.md
```

`.gitignore` adds: `data/`, `var/`, `models/*.onnx`, `ml/runs/`, `wandb/`, `*.npy`, `.venv/`.

---

# PART 6 — CONTRACTS, DATA MODEL & API

## 6.1 `AnalysisResult` v2 (the verdict contract)

`contracts/analysis.schema.json` is the source of truth; `datamodel-code-generator` → Pydantic, `json-schema-to-zod` → TS. Abbreviated shape:

```jsonc
{
  "schema_version": "2.0",
  "id": "01J…",                          // UUIDv7
  "file": { "name": "clip.mp4", "sha256": "…", "duration_s": 8.4, "assessed_s": 8.4,
            "width": 640, "height": 360, "fps": 25, "has_audio": true },
  "verdict": {
    "label": "FAKE",                      // REAL | SUSPECT | FAKE
    "p_fake": 0.93,                       // calibrated (temperature-scaled)
    "integrity": 7,                       // round(100*(1-p_fake))
    "conformal_set": ["FAKE"],
    "alpha": 0.05,
    "bucket": "gen_2_3",
    "reason": null,                       // no_speech | no_face | too_short | uncertain | branch_conflict | heavy_degradation
    "headline_en": "Lip movement stops matching the voice at 3.1–5.0 s.",
    "headline_hi": "3.1–5.0 सेकंड पर होंठ आवाज़ से मेल नहीं खाते।"
  },
  "branches": {
    "sync":   { "status": "ok", "p_fake": 0.91, "offset_ms": 320, "gate_w": 0.46,
                "curve": [/* 250 floats @25fps */], "spans": [{ "t0": 3.1, "t1": 5.0, "score": 0.94 }],
                "heatmap_ref": "artifacts/sync_heatmap.png" },
    "splice": { "status": "ok", "p_spliced": 0.12, "p_synthetic": 0.08, "gate_w": 0.31,
                "frame_spoof": [/* 50 fps */], "boundaries": [] },
    "face":   { "status": "ok", "p_fake": 0.40, "gate_w": 0.11, "frames": [/* 16 */],
                "cam_refs": ["artifacts/cam_0.png", "…"] },
    "age":    { "status": "ok", "gen_hat": 2.6, "gen_dist": [0.02,0.08,0.31,0.41,0.14,0.04],
                "platform": { "whatsapp": 0.88, "original": 0.05, "other": 0.07 },
                "why": ["audio bandwidth 9.8 kHz", "QP mean 31", "WhatsApp mux signature"] }
  },
  "timeline": { "fps": 25, "speech": [/* energy */], "mouth_open": [/* aperture */] },
  "annotation": { "status": "pending", "words": [] },   // filled by async Whisper event
  "timings_ms": { "decode": 380, "landmarks": 790, "sync": 910, "splice": 1350,
                  "face": 480, "age": 40, "fusion": 60, "total": 3860 },
  "models": { "syncformer": "2.0.3+sha:ab12", "spliceloc": "…", "facetrace": "…",
              "forwardage": "…", "nyaya": "…" },
  "mode": "live"                          // live | offline_fallback | chain_gen_k
}
```

## 6.2 SQLite schema

```sql
CREATE TABLE cases (
  id TEXT PRIMARY KEY,                 -- uuidv7
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  source TEXT NOT NULL CHECK (source IN ('upload','sample','webcam','redteam','chain')),
  title TEXT,
  expires_at TEXT NOT NULL             -- created_at + 24h (D8)
);

CREATE TABLE media (
  id TEXT PRIMARY KEY,
  case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
  kind TEXT NOT NULL CHECK (kind IN ('original','gen','attack','resync')),
  gen INTEGER CHECK (gen BETWEEN 0 AND 5),
  parent_media_id TEXT REFERENCES media(id),
  path TEXT NOT NULL, sha256 TEXT NOT NULL,
  duration_s REAL, width INTEGER, height INTEGER, has_audio INTEGER,
  emulator_params TEXT                 -- JSON (ForwardChain draw) when kind='gen'
);

CREATE TABLE analyses (
  id TEXT PRIMARY KEY,
  media_id TEXT NOT NULL REFERENCES media(id) ON DELETE CASCADE,
  status TEXT NOT NULL CHECK (status IN ('queued','running','done','error')),
  label TEXT CHECK (label IN ('REAL','SUSPECT','FAKE')),
  p_fake REAL, reason TEXT, bucket TEXT,
  result_json TEXT,                    -- full AnalysisResult v2
  error TEXT,
  total_ms INTEGER,
  model_versions TEXT NOT NULL,        -- JSON
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  finished_at TEXT
);
CREATE INDEX idx_analyses_media ON analyses(media_id);

CREATE TABLE artifacts (
  id TEXT PRIMARY KEY,
  analysis_id TEXT NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
  name TEXT NOT NULL,                  -- sync_heatmap.png, cam_0.png, dossier.pdf, card_hi.png …
  path TEXT NOT NULL, mime TEXT NOT NULL
);

CREATE TABLE chains (
  id TEXT PRIMARY KEY,
  case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
  source_media_id TEXT NOT NULL REFERENCES media(id),
  status TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE attacks (
  id TEXT PRIMARY KEY,
  case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
  kind TEXT NOT NULL CHECK (kind IN ('splice_delete','splice_swap','tts_insert','lipsync_musetalk','lipsync_latentsync')),
  params TEXT NOT NULL,                -- JSON (e.g. removed word span)
  input_media_id TEXT NOT NULL REFERENCES media(id),
  output_media_id TEXT REFERENCES media(id),
  status TEXT NOT NULL
);
```

A janitor task deletes cases past `expires_at` on server start and hourly.

## 6.3 API surface

| Method & path | Body / params | Returns |
|---|---|---|
| `POST /v1/analyses` | multipart `file` · `source` · `title?` | `202 {id, case_id, events_url}` |
| `GET /v1/analyses/{id}/events` | — | SSE stream (§6.4) |
| `GET /v1/analyses/{id}` | — | `AnalysisResult` |
| `GET /v1/analyses/{id}/artifacts/{name}` | — | file |
| `POST /v1/analyses/{id}/resync` | `{offset_ms}` | `{media_url}` — audio shifted by offset (ffmpeg `-itsoffset`), for re-sync playback |
| `GET /v1/analyses/{id}/dossier.pdf` · `/card.png?lang=hi` | — | report |
| `POST /v1/chains` | `{media_id, gens:[1..5]}` | `202 {chain_id, events_url}` — emits `gen_ready {gen, media_url, params}` then that gen's analysis events |
| `POST /v1/redteam/splice` | `{media_id, op: delete|swap|duplicate, word_indices}` or `{t0,t1}` | `{media_id, media_url, truth:{boundaries}}` — runs in < 3 s |
| `GET /v1/redteam/words/{media_id}` | — | Whisper word list with timestamps (for the word-picker UI) |
| `POST /v1/redteam/lipsync` | `{media_id, audio_source: other_media_id|tts_text, lang}` | `202 {attack_id}` — GPU job; progress via SSE |
| `GET /v1/benchmarks` | — | `ml/reports/benchmark.json` |
| `GET /v1/atlas` | — | UMAP 3D points `{id,label,gen,generator,xyz}` + model tag |
| `GET /v1/health` | — | `{ok, models:[{name,version,sha}], providers, warm:true, cpu_threads}` |

Errors: RFC 9457 `application/problem+json` with `type`, `title`, `detail` written for the user ("This file has no video stream. Upload an MP4 or WebM.").

## 6.4 SSE event types

```
event: stage       data: {"stage":"decode|landmarks|vad|codec|fusion","t_ms":380,"progress":0.12}
event: preview     data: {"kind":"mouth_strip","url":"…/mouth_strip.png"}     # early visual
event: branch      data: {"branch":"age|face|sync|splice","result":{…BranchResult}}
event: verdict     data: {…verdict, timings_ms}
event: annotation  data: {"words":[{"w":"नहीं","t0":3.38,"t1":3.61}], "boundary_context":[…]}
event: gen_ready   data: {"gen":3,"media_url":"…","params":{…}}               # chains only
event: error       data: {"type":"…","title":"…","detail":"…"}
event: end         data: {}
```

---

# PART 7 — FRONTEND & THE DESIGN SYSTEM

*Built with the `frontend-design` skill's process: design plan → review against the brief → build → critique with screenshots.*

## 7.1 Grounding

- **Subject:** forensic analysis of forwarded phone videos, in India, under time pressure.
- **Audience:** a fact-checker (and, today, a panel of engineering judges) deciding whether a clip is safe to call out.
- **Primary job of every screen:** make the *evidence* legible fast — where the mouth stops matching, where the cut is, how degraded the clip is — and then let them hand that evidence back into WhatsApp.
- **Vernacular of the subject world:** video scopes and waveform monitors, film-strip contact sheets, evidence bags with **tamper-evident tape**, SMPTE colour bars, forwarded-message chrome ("Forwarded many times").

## 7.2 Design plan (tokens)

**Concept — "the forensic light-table."** A cool, evenly lit bench where evidence (the clip) lies in the middle; instruments (scopes) dock around it on graphite panels; conclusions are applied like **tamper-evident seals**. v1's neo-brutalist paper/ink look is retired `[DECISION]`: it read as a poster, v2 must read as an instrument.

**Colour (6 base values):**

| Token | Hex | Role |
|---|---|---|
| `--bench` | `#E9EEEC` | Page surface — cool, slightly green-grey light table (deliberately not warm cream) |
| `--graphite` | `#22282B` | Instrument panels, primary text |
| `--rule` | `#B9C3C0` | Hairlines, grid on scopes |
| `--signal-sync` | `#12A6C4` | M1 SyncFormer (SMPTE cyan) |
| `--signal-splice` | `#C2338F` | M2 SpliceLoc (SMPTE magenta) |
| `--signal-face` | `#D9B21B` | M4 FaceTrace (SMPTE yellow) |

Verdict seals: `--seal-real #1D7A55`, `--seal-suspect #C57A00`, `--seal-fake #C23A2B`. ForwardAge uses a graphite→`--rule` ramp (age is context, not alarm). Dark mode: the bench becomes `#15191B`, panels `#1E2427`; signal colours unchanged (checked for ≥ 3:1 against both surfaces).

**Type:**
- **Anek Latin + Anek Devanagari** (Ek Type — an Indian foundry; one design system across Latin, Devanagari, Bangla, Tamil, Gujarati…, variable **width 75–125** and weight 100–800). One family everywhere, so Hindi and English sit on the same baseline and colour.
- **The type treatment is the concept:** headlines on the Chain Theatre and the hero literally **lose width as the clip loses generations** (`font-variation-settings: "wdth" 125 → 75` from Gen-0 to Gen-5). Compression made visible in the letters.
- Scale (1.25 ratio, 18 px body): 14.4 · 18 · 22.5 · 28 · 35 · 44 · 55. Body line-length ≤ 70 ch; line-height 1.5 (Latin) / 1.65 (Devanagari).
- Numbers: `font-variant-numeric: tabular-nums` for timings and scores.
- No all-caps labels, no tracked-out eyebrows, no mono-for-data (skill calibration list).

**Layout concept:**

```
 Desk (/)                                          Case board (/case/[id])
 ┌──────────────────────────────────────────┐      ┌──────────────────────────────────────────────┐
 │ SatyaCheck      Desk  Chain  Red team  Lab│      │ ◂ Desk   clip.mp4 · 8.4 s · Gen≈2.6 WhatsApp  │
 │                                           │      │ ┌───────────────────────┐ ┌────────────────┐ │
 │  Is this clip real?                       │      │ │                       │ │  FAKE seal     │ │
 │  (headline width = compression demo)      │      │ │   clip on the light   │ │  p=0.93 · 3.9 s│ │
 │  ┌─────────────────────────────────────┐  │      │ │   table + landmark    │ │  why (1 line)  │ │
 │  │  drop zone = empty light table       │  │      │ │   overlay; seal tape  │ │  conformal set │ │
 │  │  [ Record yourself ]                 │  │      │ └───────────────────────┘ └────────────────┘ │
 │  └─────────────────────────────────────┘  │      │ ┌ SyncScope (graphite) ──────────────────────┐│
 │  Evidence tapes:  ▣ Dubbed speech  ▣ Cut  │      │ │ curve · spans · heatmap · [Play re-synced]  ││
 │                   ▣ AI anchor  ▣ Real …   │      │ └────────────────────────────────────────────┘│
 │                                           │      │ ┌ Splice scope ─────────────┐┌ Age dial ────┐│
 │  live stage rail (appears on run):        │      │ │ waveform · pins · words   ││ Gen 0..5     ││
 │  decode ▸ landmarks ▸ age ▸ face ▸ sync ▸ │      │ └───────────────────────────┘└──────────────┘│
 │  splice ▸ verdict   (real ms each)        │      │ ┌ Face trace (3D mesh heat + CAM strip) ─────┐│
 └──────────────────────────────────────────┘      │ └────────────────────────────────────────────┘│
                                                    │  [ Download dossier ]  [ Share card (हिंदी) ] │
                                                    └──────────────────────────────────────────────┘
```

Alignment: left-aligned text everywhere; instruments on a 12-col grid (clip 7 cols, verdict 5; scopes full width). Mobile: single column; clip → seal → scopes.

**Principles (what makes it this product and no other):**
1. **One bold thing per screen.** Desk: the width-compressing headline. Case: the seal tape across the clip. Chain: the six-frame contact sheet. Lab: the 3D atlas. Everything else is quiet graphite and hairlines.
2. **Every colour is a branch.** Cyan is always sync, magenta always splice, yellow always face — on scopes, legends, the pipeline rail and the Lab. No decorative colour.
3. **Numbers are measured or absent.** Every figure on screen comes from an API field; nothing hard-coded (A8).
4. **Motion only answers something.** The seal applies once when the verdict lands; stage rail ticks as SSE events arrive; the Chain filmstrip advances as each gen is ready. No scroll-triggered fade-ins.

**Review against the brief (skill step):** a first draft used near-black panels with one acid-green accent — that is calibration trait #2, so the page moved to a light bench with *categorical* signal colours tied to branches. A rubber-stamp verdict (v1) was replaced by the evidence-tape seal, which belongs to the forensic subject, not to "poster".

## 7.3 The screens

### 7.3.1 Desk (`/`)
- Hero = the instrument itself: headline "Is this clip real?" (en/hi toggle), drop zone, **Record yourself** (webcam + mic, 8 s countdown, MediaRecorder → webm), and **evidence tapes** (sample cards styled as labelled tape cassettes): *Dubbed speech* (MuseTalk fake, has audio), *One word cut* (splice), *AI anchor* (v1, mute), *Real man* (v1, mute), *Real speech* (clean control), *Forwarded 5×* (Gen-5 of the dubbed clip).
- On run: a **stage rail** fills left→right, each node showing its real ms; branch nodes coloured by branch; then navigates to `/case/[id]` (or morphs in place on desktop).

### 7.3.2 Case board (`/case/[id]`)
- **Clip on the light table** with toggles: landmark mesh overlay (MediaPipe in browser), mouth ROI zoom (the 88×88 the model saw, from `mouth_strip.png`).
- **Seal:** tamper-evident tape band diagonally across the clip frame's corner (`REAL` / `SUSPECT` / `FAKE` repeated along the tape, in Anek condensed), `p_fake`, total seconds, one-line headline, conformal set chips (`{FAKE}`), reason if SUSPECT.
- **SyncScope** (cyan): speech energy + mouth aperture traces, sync-error curve, dubbed spans, the 31-row AV offset heatmap under it, and **[Play re-synced]** (calls `/resync`).
- **Splice scope** (magenta): wavesurfer waveform, frame-spoof heat strip under it, boundary pins; word chips from Whisper with the cut marked `✂`.
- **Age dial** (graphite): Gen 0…5 distribution bars, platform, the three "why" features.
- **Face trace** (yellow): 4 Grad-CAM frames + R3F face mesh (468 vertices from the landmarks, per-vertex heat from CAM sampled at vertex UVs), orbitable.
- **Decision trace** (collapsible): gate weights per branch as a stacked bar, bucket, α, set.
- Actions: **Download dossier (PDF)**, **Share card** (1080×1350 PNG, en or hi, sized for WhatsApp), **Forward this 5×** (→ Chain Theatre with this media).

### 7.3.3 ForwardChain Theatre (`/chain`)
- Top: a **contact sheet of six frames** Gen-0…Gen-5 (the same frame from each generation), filling in live as `gen_ready` events arrive; each thumbnail shows resolution, kbps and a zoomed 64×64 mouth crop so blockiness is visible.
- Below: **the verdict line chart** — x = generation, y = p_fake — SatyaCheck line (graphite) vs **pixel-only baseline** (dashed) vs `baseline-v1` (dotted); the baseline's collapse is the point.
- The page headline width compresses with the currently displayed gen (the typographic concept at full volume).
- Toggle "Real WhatsApp" shows T-WA pairs (emulated vs truly forwarded) side by side with their ffprobe numbers — the calibration proof.

### 7.3.4 Red team (`/redteam`)
- Step 1: record the judge (or pick a sample). Whisper returns word chips.
- Step 2: tap a word to delete / drag to swap → **Make the fake** (splice, < 3 s) → plays the tampered clip ("the judge now says the opposite").
- Step 3: **Scan it** → case board opens; the pin lands on the cut.
- Lip-sync card: pre-generated MuseTalk fakes of teammates + "generate new" (GPU, 1–3 min, progress bar) `[CUT for live generation]`.

### 7.3.5 Lab (`/lab`)
- **Model X-ray:** the architecture as an interactive diagram (Part 4 figures redrawn in SVG), each block showing params, input shape, latency.
- **Benchmark Wall:** renders `benchmark.json`: ROC per test set, AUC-vs-generation, per-language table, ablations, baselines, reliability diagram, latency histogram, conformal coverage per bucket. Footer: model hashes + "reproduce: `make eval`".
- **Embedding atlas (3D):** UMAP of SyncFormer clip embeddings for ~1,500 test clips as an R3F point cloud (colour = real/fake, shape = generator, size = gen); the last analysed clip flies in and lands in its neighbourhood.

## 7.4 The dossier (practical-impact artefact)

- PDF (A4) and a share card (PNG 1080×1350). Bilingual (en + hi; mr in `[CUT]`).
- Contents: verdict + calibrated confidence in plain words ("very likely edited"), 3 key frames, the timeline strip with spans/pins, the cut words, generation estimate, what the tool cannot tell ("we cannot identify who made it"), SHA-256 of the analysed file, model versions, timestamp, a QR to the case URL.
- Tone: advisory, never accusatory ("This clip shows signs of editing" not "This is fake news").

## 7.5 States

| State | Treatment |
|---|---|
| Empty desk | The drop zone *is* the invitation; tapes below; no illustration. |
| Running | Stage rail with live ms; the clip shows landmark overlay as soon as `preview` arrives. |
| SUSPECT | Amber seal + the reason as a sentence + "what would resolve it" (e.g. "a version with sound", "a clip where the face is visible"). |
| Error | Specific and actionable (problem+json `detail`), with a retry and "try a sample". |
| API unreachable | Banner "Offline mode — running the on-device checks only" and the ORT-web tier result (clearly labelled `mode: offline_fallback`). |
| Reduced motion | Seal appears without tape animation; charts render final state. |

## 7.6 Component inventory

`ui/`: Button, IconButton, Toggle, Tabs, Tooltip, Sheet, Toast (built on Radix primitives where needed — `@radix-ui/react-*`).
`desk/`: DropTable, RecordYourself, EvidenceTape, StageRail.
`case/`: ClipTable (video + overlays), Seal, SyncScope, OffsetHeatmap, SpliceScope, WordChips, AgeDial, FaceTrace, DecisionTrace, DossierActions.
`chain/`: ContactSheet, GenerationChart, CalibrationPairs.
`redteam/`: WordPicker, AttackPlayer, LipsyncQueue.
`lab/`: ArchDiagram, BenchmarkWall, ROCPlot, ReliabilityPlot, LatencyHist.
`three/`: FaceMeshHeat, EmbeddingAtlas (instanced points, drei `OrbitControls`, `Bounds`).

Quality floor: responsive to 360 px, visible focus rings, keyboard for every control, `prefers-reduced-motion`, WCAG AA contrast, `lang="hi"` on Devanagari runs.

---

# PART 8 — THE DEMO EXPERIENCES & THE PITCH

## 8.1 Samples to ship (replacing "all three are mute")

| Tape | Content | Expected | Proves |
|---|---|---|---|
| Real speech | Teammate, Hindi, 8 s, phone camera | REAL | Clean control |
| Dubbed speech | Same teammate, MuseTalk-dubbed with a different sentence | FAKE (sync spans) | Lip-sync detection (A2) |
| One word cut | Teammate: "मैं इस फैसले का समर्थन **नहीं** करता" → "नहीं" removed | FAKE (splice pin at the cut) | Splice localisation (A3) — the riot scenario |
| Forwarded 5× | Dubbed speech after ForwardChain Gen-5 (and a true 5-hop WhatsApp copy) | FAKE, gen≈5 | The twist (A4) |
| AI anchor (v1) | Fully generated, mute | FAKE via FaceTrace, sync/splice abstain `no_speech` | Visual branch + honest abstention |
| Real man (v1) | Mute real | SUSPECT `no_speech` | Abstention (A6) |
| Held-out LatentSync | Unseen generator | FAKE (report honestly if not) | Generalisation |

All consented recordings of team members; clips ≤ 2 MB each.

## 8.2 The six "wow" moments

1. **Fake the judge (Red team).** Record a judge saying a sentence with a negation. Remove one word. Play it back — they now say the opposite. Scan: the pin lands on the cut, Whisper names the removed position. *This is the PS scenario, performed on a judge, in 20 seconds.*
2. **Watch the lips snap back (SyncScope re-sync).** On a delay-type dub, press *Play re-synced* — audio shifts by the model's predicted offset and the mouth visibly locks. The model didn't just say "fake", it said *by how much*.
3. **Forward it five times, live (Chain Theatre).** Contact sheet degrades frame by frame, the headline type condenses, our line stays flat while the pixel-only baseline collapses.
4. **The WhatsApp we measured.** Emulated Gen-5 next to a real 5-hop WhatsApp forward, same numbers. "We didn't guess what WhatsApp does. We forwarded 36 clips 180 times."
5. **It knows when it doesn't know.** Mute real clip → SUSPECT with a reason; conformal coverage per bucket on the Lab wall.
6. **Reproduce us.** Lab → Benchmark Wall → "every number here comes from `make eval`, model hashes in the footer."

## 8.3 Five-minute pitch script (with a demo driver and a narrator)

| Time | Beat | On screen | Owner |
|---|---|---|---|
| 0:00–0:30 | The riot: a 6-second clip, one word removed, a town on fire. "Fact-checkers get 10 seconds and a laptop." | Desk, headline condensing | Narrator |
| 0:30–1:30 | **Fake the judge** (wow 1). | Red team → Case board | Driver |
| 1:30–2:10 | How it works in one breath: four models, trained on *real* people, *through* WhatsApp. Name the pieces: SyncFormer, SpliceLoc, FaceTrace, ForwardAge, Nyaya. | Lab → Model X-ray | Narrator |
| 2:10–2:50 | **Lip-sync dub + re-sync** (wow 2). | Case board | Driver |
| 2:50–3:40 | **Forward it five times** (wow 3) + **the WhatsApp we measured** (wow 4). | Chain Theatre | Driver + narrator |
| 3:40–4:20 | Honesty: abstention, conformal guarantee, Benchmark Wall numbers incl. a weakness. | Lab | Narrator |
| 4:20–4:50 | Impact & scale: dossier in Hindi forwarded back into the group; ₹0.02/clip; tipline design. | Dossier card | Narrator |
| 4:50–5:00 | Team: who built which model. | Team slide | All |

## 8.4 Judge Q&A — prepared answers

| Likely question | Answer (short) |
|---|---|
| "Did you train anything or just use pretrained models?" | We trained four models (SyncFormer, SpliceLoc, FaceTrace, ForwardAge) and the fusion. Pretrained AV-HuBERT, AVH-Align and W2V2-AASIST are *teachers* we distil from — they are too big for a 10 s CPU budget. Show the W&B project (curves, ROC, confusion matrices). |
| "How do you know it works on WhatsApp and not just your emulator?" | T-WA: real 5-hop forwards; numbers side by side on the Chain page. |
| "What about generators you've never seen?" | T-HELD (LatentSync) — reported separately; real-only pre-training is the reason it transfers. |
| "Your dataset is small." | Yes — which is why the representation comes from real-only self-supervision and teachers trained on thousands of hours; our labelled data only fine-tunes heads. Splice data is generated on the fly (unlimited). |
| "Isn't a mute clip just 'fake'?" | No — mute means unverifiable. We abstain with a reason. The AI anchor is caught by FaceTrace, not by being mute. |
| "Why not an LLM?" | Hallucinated explanations in a misinformation tool are a harm. Explanations are templated from measured evidence. |
| "Data leakage?" | Identity-disjoint splits; leading-silence trimmed (AVH-Align shortcut finding). |
| "Latency on a weaker laptop?" | p95 measured with GPU hidden and 4 threads; budget guard; live timings on every run. |
| "Accuracy numbers?" | Read them off the Benchmark Wall — never from memory. |


---

# PART 9 — THE BUILD PLAN

## 9.0 Read this first: time, the cut line, and who does what

**Calendar reality `[RISK]`.** The site lists the Grand Finale for **27 Sep 2026, 09:00 IST** and says finalists "build live". This document was written at 00:30 on 27 Sep. Confirm the actual judging slot before you start. Two tracks follow:

- **Full plan** (P0–P10, ≈ 41 engineer-hours + background GPU time) — for when you have a day or more, or four people working in parallel.
- **Finale fast-path** (§9.12, ≈ 12 engineer-hours across 4 people ≈ 3–4 wall-clock hours each + ~3 h background training) — a strict subset that still delivers trained models, measured numbers, the Chain Theatre and the red-team demo. **Everything marked `[CUT]` is outside the fast-path.**

**Ownership (mirrors the Round-1 deck roles, one Claude session each):**

| Member | Owns | Phases |
|---|---|---|
| M1 — ML visual | SyncFormer, FaceTrace, teacher features | P2.3, P4, P5.1 |
| M2 — Audio + fusion | SpliceLoc, Nyaya, splice factory | P1.5, P3, P6 |
| M3 — Backend, simulator, eval | ForwardChain, preprocess, server, eval harness | P0, P1.1–1.4, P2.1–2.2, P5.2, P7, P8 |
| M4 — Frontend, demo, pitch | Web app, dossier, samples, rehearsal | P1.6 (MuseTalk samples), P9, P10.4–10.5 |

**Standing rules for every phase:**
1. Every sub-phase ends with a commit: `git commit -m "P3.2: SpliceLoc model + heads"`. A fresh session reads `git log --oneline -20` + `PROGRESS.md`, never the old chat.
2. Update `PROGRESS.md` (one line per sub-phase: ✅/🟡/❌ + the key number). It is the cheapest context a new session can load.
3. Long jobs (downloads, ladders, training, extraction) run with `nohup … > logs/X.log 2>&1 &`. Check them with `tail -n 30`, never `cat`.
4. No number enters the UI or README unless it came from `benchmark.json` (§4.11).
5. **Do not touch** `vercel.json`, the Vercel project settings, or `lib/analyze.ts` (it becomes `baseline-v1` as-is).

---

## P0 — Foundations · 1.5 h · owner M3

| | Sub-phase | Model | Detail | Done when |
|---|---|---|---|---|
| **P0.1** | Python toolchain · 20 min | 🟢 Haiku | `uv python install 3.11` · `uv venv -p 3.11 .venv` · `pyproject.toml` with deps from §12.1 · `uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128` (Blackwell sm_120 needs cu128) · verify `python -c "import torch;print(torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"` shows `sm_120`. | RTX 5050 visible to torch; `import onnxruntime, mediapipe, av` works |
| **P0.2** | Repo layout · 15 min | 🟢 Haiku | Create the folders of §5.7 with `__init__.py`; extend `.gitignore` (§5.7); `Makefile` with the targets of §12.3 (stubs OK); `PROGRESS.md` with every sub-phase ID from this Part. | `make help` lists targets |
| **P0.3** | Weights download (background) · 15 min | 🟢 Haiku | Run `ml/data/download/weights.sh` (§12.2): XLS-R-300M (HF), W2V2-AASIST, AV-HuBERT `self_large_vox_433h.pt`, AVH-Align checkpoint, MediaPipe `face_landmarker.task`, silero-vad, faster-whisper small, SBI EfficientNet weights, MuseTalk 1.5, LatentSync 1.5 `[CUT]`. Into `~/.cache/satya/weights/` with a `SHA256SUMS`. | All files present; sums recorded |
| **P0.4** | Datasets (background) · 20 min | 🟢 Haiku | Open sets via `ml/data/download/datasets.sh`: IndicTTS-Deepfake-Challenge (HF), IndicSynth **hi + mr subsets only**, PartialSpoof (Zenodo), AVLips (Drive), LipSyncTIMIT, InDeepFake. Submit licence forms for FakeAVCeleb, AV-Deepfake1M, HAV-DF (nothing depends on them). | Downloads running; forms sent |
| **P0.5** | Contract · 20 min | 🔵 Sonnet | Write `contracts/analysis.schema.json` from §6.1; generate `server/schemas.py` (datamodel-code-generator) and `lib/contracts.ts` (json-schema-to-zod); `tests/test_contract.py` validates a fixture. | Both codegens run from `make contracts` |
| **P0.6** | Human: recording session kickoff · parallel | — | M4 books 45 min with 6–10 consenting people: 3 scripted sentences each in Hindi/Marathi/English (one containing a negation), 8–12 s, phone front camera, quiet + noisy room. Consent line recorded on camera. | ≥ 60 clips in `data/raw/own/` |

**Exit:** env works on GPU and CPU; downloads streaming; contract compiles both sides.

---

## P1 — The data engine · 5 h · owners M3 (P1.1–1.4), M2 (P1.5), M4 (P1.6)

| | Sub-phase | Model | Detail | Done when |
|---|---|---|---|---|
| **P1.1** | Manifests & identity-disjoint splits · 1 h | 🔵 Sonnet | `ml/data/manifests.py`: one parquet row per clip `{clip_id, path, dataset, speaker_id, lang, label, generator, has_audio, duration, fake_spans}`. Speaker IDs from dataset metadata (VidTIMIT speaker folders, AVLips source IDs, our recording sessions). `splits.py`: 70/10/10/10 by speaker, stratified by lang & generator; **hold out LatentSync + IP_LAP entirely** (T-HELD). Write `data/manifests/{all,train,val,cal,test}.parquet`. Unit test: no speaker in two splits. | `pytest ml/tests/test_splits.py` green; split table printed per lang/generator |
| **P1.2** | ForwardChain emulator (priors) · 1 h | 🔵 Sonnet | `ml/forwardchain/emulate.py` implementing §4.2.2 `hop()` with prior distributions (long side ∈ {640, 848, 480}, CRF U[28,34], x264 preset/profile draws, VFR prob 0.3, AAC 48–96 kbps, mono prob 0.4, screen-rerecord 0.08, overlay 0.05). CLI: `python -m ml.forwardchain.emulate in.mp4 --gens 5 --out dir/ --seed 7` → `gen_1.mp4 … gen_5.mp4` + `params.json`. Deterministic per seed. | 5 gens produced in < 3 s for a 10 s clip; ffprobe shows decreasing bitrate/bandwidth |
| **P1.3** | WhatsApp calibration · 1.5 h (mostly phone time) | 🔵 Sonnet | Humans: forward 36 source clips × 5 hops (§4.2.2 protocol), download via WhatsApp Web into `data/raw/whatsapp/{clip}/{hop}.mp4`. Code: `measure.py` (ffprobe → table), `fit.py` (SNVSE CRF search per pair → `p(CRF|res,hop)`, resize rule, AAC bitrate, VFR prob) → `whatsapp_2026.json`; emulator switches from priors to the fitted file. Keep 8 source clips out as **T-WA** test. | JSON written; plot `ml/reports/wa_calibration.png` (emulated vs real bitrate per hop) |
| **P1.4** | Generate ladders (background) · 15 min setup + ~2 h bg | 🟢 Haiku | `ml/forwardchain/ladder.py` over train/val/cal/test manifests with a 12-process pool; writes `data/gens/{clip_id}/gen_{k}.mp4` + `data/manifests/gens.parquet`. Skip existing (resumable). | Row count = clips × 6; spot-check 5 random gens play |
| **P1.5** | Splice factory · 1 h | 🔵 Sonnet | `ml/redteam/splice.py`: faster-whisper (small, int8, `word_timestamps=True`, `language` auto) → words; ops `delete`, `swap`, `duplicate`, `cross_take`; crossfade 0–15 ms, loudness match (pyloudnorm), optional room-tone fill; returns audio + `boundaries[]` + frame labels @20 ms. Also exposed as a torch `IterableDataset` for M2 (on-the-fly). `tts_insert.py` `[CUT]`: XTTS-v2 word insertion. | 20 generated splices audibly correct; boundaries match the cut sample to ±10 ms |
| **P1.6** | MuseTalk fakes (GPU, background) · 1 h setup + 1–3 min/clip bg | 🔵 Sonnet → 🟢 Haiku | Separate env `envs/musetalk` (its own Python 3.10 + pinned torch; if its torch pin lacks sm_120, install torch 2.7 cu128 and patch — budget 30 min, else use Wav2Lip). `musetalk_batch.py`: for each own-recording speaker, dub (a) their re-ordered audio, (b) another speaker's sentence; also **partial dubs** (1.5–3 s window). Target 150–300 fakes + the demo tapes of §8.1. LatentSync for T-HELD `[CUT]`. | Fakes in `data/raw/redteam/`, rows appended to manifest with `generator=musetalk` |

**Exit:** manifests + splits frozen (hash recorded in `PROGRESS.md`); ForwardChain calibrated; ladders generating; splice factory working; first MuseTalk fakes exist.

---

## P2 — Preprocessing & feature caches · 3.5 h · owner M3 (+ M1 for P2.3)

| | Sub-phase | Model | Detail | Done when |
|---|---|---|---|---|
| **P2.1** | `satya_core.preprocess` · 1.5 h | 🔵 Sonnet | Implement §4.2.4 exactly: PyAV decode → 25 fps CFR frames + 16 kHz mono; MediaPipe Face Landmarker (VIDEO mode, `num_faces=2`, pick largest with IoU tracking, every 2nd frame + linear interpolation of landmarks); similarity alignment to a mean face (store `ml/assets/mean_face_478.npy`); mouth ROI 96→88 grayscale; 224 face crops; 8-d kinematics; silero VAD speech ratio; abstain flags. Return a typed `Preprocessed` dataclass. | `preprocess(clip)` on a 10 s clip < 1.5 s CPU; visual check of 20 mouth strips saved as PNG |
| **P2.2** | Codec features · 45 min | 🔵 Sonnet | `features_codec.py`: ffprobe JSON → stream features; PyAV with `flags2=+export_mvs` → MV histograms; per-frame QP via `ffmpeg -debug qp` parse (fallback: estimate from I-frame DCT); Benford first-digit of quantised 8×8 DCT (I-frames, luma); blockiness; Laplacian var; AAC bitrate + spectral rolloff (−60 dB bandwidth). 180-d vector + names. | Features vary monotonically-ish with gen on 20 ladders (plot) |
| **P2.3** | Teacher features · 1 h setup + ~40 min bg | 🟣 **Opus** (env debugging) → 🟢 Haiku (run) | Separate env `envs/teacher` for AV-HuBERT (fairseq). **Fallback ladder (time-box each):** ① AV-HuBERT-L + AVH-Align via their repo's env (45 min max) → ② Auto-AVSR visual encoder features (pure PyTorch, `mpc001/Visual_Speech_Recognition_for_Multiple_Languages`) (30 min) → ③ skip KD (λ_kd = 0) and record it as an ablation. Extract per-frame V/A features (fp16 .npy) + AVH-Align per-frame scores for R-AV at gen 0 and one random gen. Also cache W2V2-AASIST utterance logits for M2 data. | `data/cache/teacher/*.npy` count = expected; decision (①/②/③) written to `PROGRESS.md` |
| **P2.4** | Cache builder + parity · 30 min | 🔵 Sonnet | `ml/data/build_cache.py`: run `preprocess` over all clips × gens in a process pool → `data/cache/{clip}/{gen}/{roi.npy, mel.npy, kin.npy, face.npy, codec.npy, meta.json}`. Parity test: the server path and the cache path give byte-identical ROI for 10 clips. | Cache complete; `test_preprocess_parity` green |

**Exit:** every clip × gen has cached inputs; teacher route decided and cached.

---

## P3 — M2 SpliceLoc · 4 h (≈ 1.5 h hands-on + 2–3 h GPU) · owner M2

| | Sub-phase | Model | Detail | Done when |
|---|---|---|---|---|
| **P3.1** | Data module · 1 h | 🔵 Sonnet | Lightning `SpliceDataModule`: mixes on-the-fly splices (P1.5), TTS/partial sets (IndicTTS, IndicSynth, PartialSpoof with its segment labels resampled to 20 ms), real; applies ForwardChain audio chain at random gen (AAC re-encode in-process via PyAV — pre-encode 5 variants per clip to keep the loader fast) + RawBoost (p 0.5); 6 s random crops; silence trimmed; labels: frame (50 fps), boundary (dilated ±2), utterance, teacher logit. | 1 batch visualised: waveform + frame + boundary labels line up |
| **P3.2** | Model + losses · 1 h | 🟣 **Opus** | `ml/train/spliceloc.py`: HF `Wav2Vec2Model` from `facebook/wav2vec2-xls-r-300m`, keep `encoder.layers[:8]`, learnable weighted sum of layers 4–8; projection 1024→256; BAM Boundary-Enhancement (1D-ResNet intra + frame-wise attention inter) → boundary head; 2× Boundary-aware Frame Attention (mask from predicted boundaries, straight-through at train time using GT boundaries with p 0.5 for curriculum) → frame head; attentive-stats-pool utterance head. Loss per §4.4.2. Freeze schedule + LLRD. Unit test: forward pass shape on 6 s and 10 s. | Overfits 32 samples to ~0 loss in 200 steps (sanity) |
| **P3.3** | Train (background) · 2–3 h GPU | 🟢 Haiku | `python -m ml.train.spliceloc +exp=main` (bf16, batch 16, grad-ckpt, 8 epochs). W&B logging (`wandb.init(project="satyacheck", group=<model>)`); checkpoint on val frame-EER. Haiku checks `tail -n 30 logs/spliceloc.log` at most every 30 min and reports only EER/loss. | Val frame-EER and boundary-F1 plateau; best ckpt saved |
| **P3.4** | Eval & thresholds · 30 min | 🔵 Sonnet | Val: τ for boundary peaks (maximise F1@±80 ms), NMS gap; test by gen & language → partial `benchmark.json` entries. | Numbers in `ml/reports/spliceloc.json` |

---

## P4 — M1 SyncFormer · 5 h (≈ 2.5 h hands-on + 3 h GPU) · owner M1

| | Sub-phase | Model | Detail | Done when |
|---|---|---|---|---|
| **P4.1** | Data module · 1 h | 🔵 Sonnet | Loads cached ROI/mel/kin/teacher; random 6–10 s crop (aligned 4 mel frames : 1 video frame); per-16-frame-segment audio shift s ∈ [−15,15] (roll within the crop with edge padding, not wrap); 20% swapped-speaker audio; ROI jitter, brightness/contrast, 10% frame drop + repeat; labels: shift class, (fine-tune) frame fake labels from `fake_spans`. | A batch rendered as a GIF (ROI) + mel with shift annotations |
| **P4.2** | Model + losses · 1.5 h | 🟣 **Opus** | `ml/train/syncformer.py` exactly per §4.3.1: 3D stem + ResNet-18 2D trunk (timm `resnet18`, in_chans=1, stem replaced), audio conv-subsample + 4-layer Conformer-lite, kinematics MLP, 4-layer cross-modal transformer with local attention mask (±32), heads S/Δ/F/D; InfoNCE (τ 0.07, ±15 neighbours), shift CE (32 classes incl. no-sync), cosine KD to teacher (projection heads), focal BCE for F. Grad checkpointing on the trunk. Unit tests for shapes and for "shift=0 batch → S loss < shuffled batch". | Overfit sanity passes; params ≈ 11–13 M printed |
| **P4.3** | Stage B pre-train (background) · 2–3 h GPU | 🟢 Haiku | Real-only, all gens, `+exp=pretrain` (AdamW 3e-4, bf16, batch 24, 30 epochs cap, early stop on val shift-accuracy). | Val shift top-1 accuracy (±1 frame) reported; ckpt saved |
| **P4.4** | Stage C fine-tune · 40 min GPU | 🔵 Sonnet (launch) → 🟢 Haiku (watch) | Balanced F-LS + R-AV, freeze stem/trunk, train transformer top-2 + heads, keep shift loss λ 0.2. | Val AUC (video) + span AP recorded |
| **P4.5** | Eval · 30 min | 🔵 Sonnet | T-LS by gen & generator, T-HELD, span AP@0.5, offset error on synthetic delays (MAE ms) → `ml/reports/syncformer.json`; UMAP embeddings for the atlas → `atlas.json`. | JSON + plots written |

---

## P5 — M4 FaceTrace + M3 ForwardAge · 3 h · owners M1 (P5.1), M3 (P5.2)

| | Sub-phase | Model | Detail | Done when |
|---|---|---|---|---|
| **P5.1** | FaceTrace · 1.5 h (+1 h GPU bg) | 🔵 Sonnet → 🟢 Haiku | Port SBI's blending (landmark-hull masks, colour/frequency jitter) to use our 478-pt landmarks; region variants face / lower-face / lips; apply to *decoded gen-k frames*. timm `efficientnet_b0`, 224, AdamW 1e-4, 12 epochs. Add fully-synthetic subsets as positives. Grad-CAM (`pytorch-grad-cam`) util. | Val AUC by gen; CAM PNGs look face-centred |
| **P5.2** | ForwardAge · 1 h | 🔵 Sonnet | LightGBM ordinal (5 × binary gen>k) + platform head on `codec.npy` over the ladders; 5-fold CV grouped by source clip; SHAP top-k per prediction. Eval on T-WA (real forwards!) — MAE in generations. | MAE (emulated) and MAE (T-WA real) in `forwardage.json` |

---

## P6 — M5 Nyaya · 1.5 h · owner M2

| | Sub-phase | Model | Detail | Done when |
|---|---|---|---|---|
| **P6.1** | Branch outputs on val/cal/test · 20 min | 🟢 Haiku | Run M1/M2/M4/M3 (PyTorch) over val, cal, test → `data/cache/branch_outputs.parquet` (one row per clip×gen). | Parquet written |
| **P6.2** | Fusion + temperature + conformal · 1 h | 🟣 **Opus** | `ml/train/nyaya.py` per §4.8: gated fusion trained on val (masking unavailable branches), temperature on val, Mondrian split-conformal on cal by gen bucket (α 0.05), reason-code logic; export gate+MLP to ONNX, `T` and `q̂` to registry. Tests: empirical coverage on test ≥ 1−α−0.02 per bucket; mute clips → `no_speech`. | `nyaya.json` with ECE before/after, coverage, abstention rate per bucket |
| **P6.3** | Ablation hooks · 10 min | 🔵 Sonnet | Flags for mean-fusion and fixed-threshold variants so P7 can run them. | Flags work |

---

## P7 — Evaluation harness & benchmark · 3 h · owner M3

| | Sub-phase | Model | Detail | Done when |
|---|---|---|---|---|
| **P7.1** | `run_eval.py` · 1 h | 🔵 Sonnet | Runs the **ONNX pipeline end-to-end** (not PyTorch) over every test set in §4.10, with 6–10 s windows, producing per-clip rows + aggregate metrics (sklearn: roc_auc, average_precision, EER via ROC; boundary F1@±80 ms; span AP; ECE; coverage). Writes `benchmark.json` with model hashes + git SHA + date. | `make eval` runs clean in < 30 min |
| **P7.2** | Baselines · 1 h | 🔵 Sonnet | `baseline-v1` (Python port of `lib/analyze.ts` scoring **or** Playwright driving the v1 page headless — port is faster), AVH-Align zero-shot, W2V2-AASIST zero-shot, pixel-only EfficientNet-B0 trained on gen-0 only (20 min GPU), LIPINC-V2 / LipFD `[CUT]`. Same windows and sets. | Baseline rows in `benchmark.json` |
| **P7.3** | Ablations · 30 min (+ reuse) | 🟢 Haiku | Run the six ablation variants (§4.10) — most reuse checkpoints (fusion flags; Gen-0-only FaceTrace from P7.2); the two needing retraining (no-ForwardChain M1, no-shift M1) are `[CUT]` unless GPU is idle. | Ablation table filled (or marked not-run) |
| **P7.4** | Plots + claim-guard review · 30 min | 🔵 Sonnet → 🟣 Opus (review only, 10 min) | `plots.py`: ROC, AUC-vs-gen, reliability, latency hist, coverage bars (PNG + the JSON the web renders). Opus reads **only** `benchmark.json` + §4.11 and lists every sentence we are *allowed* to say. Save as `ml/reports/claims.md`. | `claims.md` exists; pitch uses only it |

---

## P8 — Export & the inference server · 4 h · owner M3

| | Sub-phase | Model | Detail | Done when |
|---|---|---|---|---|
| **P8.1** | ONNX export + INT8 + parity · 1.5 h | 🔵 Sonnet (🟣 Opus only if parity fails twice) | `ml/export/to_onnx.py` per §4.12 (dynamo export, opset 18, dynamic time axis; for XLS-R export the truncated encoder+heads as one graph); `quantize.py` dynamic INT8 per-channel; `parity.py` on 200 val clips (fp32 diff < 1e-2, INT8 AUC delta < 0.5). Write `models/registry.json` with sha256. Upload to HF model repo `[CUT]` (local copy is enough on the stage laptop). | Parity report green; total INT8 size ≈ 150 MB |
| **P8.2** | `satya_core` runtime · 1 h | 🔵 Sonnet | ORT session wrappers (CPU EP, `intra_op_num_threads=2`, sessions created once and warmed at startup with a dummy clip); `pipeline.analyze(path, on_event)` with ThreadPool fan-out, budget guard (§4.7), Nyaya, explain templates (en + hi strings from `satya_core/i18n/*.json`), artifact writers (mouth strip, offset heatmap, CAM PNGs). | `python -m satya_core.cli sample.mp4` prints a valid `AnalysisResult` in < 5 s with GPU hidden |
| **P8.3** | FastAPI + SSE + SQLite · 1.5 h | 🔵 Sonnet | Routes of §6.3 (analyses, events, result, artifacts, resync, health, benchmarks, atlas), `sse-starlette`, SQLModel tables of §6.2, upload validation (python-magic + ffprobe), janitor, CORS for `localhost:3000` + the Vercel origin, problem+json errors. Async Whisper annotation after verdict (faster-whisper small int8, CPU). | `tests/test_api.py` (httpx) green; SSE visible with `curl -N` |
| **P8.4** | Chain + red-team endpoints · 1 h | 🔵 Sonnet | `/v1/chains`: runs emulator gens 1..5 sequentially, emits `gen_ready`, then pipes each gen through `pipeline.analyze` (reuse sessions) — total ≈ 5 × 4 s; `/v1/redteam/words`, `/v1/redteam/splice` (P1.5 code); `/v1/redteam/lipsync` queue to `workers/musetalk_worker.py` (subprocess in `envs/musetalk`) `[CUT for live]`. | Chain of a 8 s clip completes < 30 s; splice < 3 s |
| **P8.5** | Dossier · 45 min | 🔵 Sonnet (🟢 Haiku for the Hindi copy pass) | WeasyPrint HTML template (Anek via local font files) → PDF; Pillow share card 1080×1350 en/hi with key frame, seal, timeline strip, QR (`qrcode`). | PDF + PNG for 3 sample cases look right |

---

## P9 — Frontend v2 · 8 h · owner M4

*Before writing any Next.js code: read `node_modules/next/dist/docs/` for anything unfamiliar — this repo's `AGENTS.md` warns Next 16 differs from training data.*

| | Sub-phase | Model | Detail | Done when |
|---|---|---|---|---|
| **P9.1** | Shell, tokens, fonts, i18n · 1 h | 🔵 Sonnet | `globals.css` tokens of §7.2 (light + dark); Anek Latin + Anek Devanagari via `next/font/google` with `axes: ["wdth"]`; `next-intl` en/hi; top nav (Desk, Chain, Red team, Lab); `lib/api.ts` (base from `/api-base.json` → env → `http://localhost:8000`), `lib/sse.ts` (EventSource wrapper with typed events via zod), `lib/store.ts` (zustand analysis state machine: idle→uploading→streaming→done|error|offline). | Shell renders at 360 px and 1440 px; `api-base.json` swap works without rebuild |
| **P9.2** | Desk · 1.5 h | 🔵 Sonnet | DropTable, EvidenceTape (samples of §8.1), RecordYourself (getUserMedia, 8 s countdown, MediaRecorder webm/opus, preview, consent line), StageRail driven by SSE `stage`/`branch` events with real ms; width-compressing headline (CSS `font-variation-settings` bound to a `--gen` custom property). | Upload → rail fills → navigates to case |
| **P9.3** | Case board · 2.5 h | 🔵 Sonnet | ClipTable (video + MediaPipe landmark overlay canvas + mouth-ROI inset), Seal (tape SVG pattern, one-shot motion), SyncScope (d3 paths: speech, aperture, sync curve, spans) + OffsetHeatmap (canvas 31×N), Play re-synced (`/resync`), SpliceScope (wavesurfer + regions for boundaries + spoof heat canvas) + WordChips (annotation event), AgeDial, DecisionTrace, DossierActions. | All branches render from a recorded SSE fixture (`tests/fixtures/case_fake.jsonl`) |
| **P9.4** | 3D · 1.5 h | 🔵 Sonnet | `three/FaceMeshHeat`: build a `BufferGeometry` from the 468 landmark points + MediaPipe's canonical triangulation (`FACEMESH_TESSELATION` indices), per-vertex colour from CAM sampled at each landmark's 2D position; slow auto-rotate off by default, OrbitControls. `three/EmbeddingAtlas`: `InstancedMesh` of `atlas.json` points, the current clip animates in. Lazy-load both (`next/dynamic`, `ssr:false`). | 60 fps on the stage laptop; no SSR errors |
| **P9.5** | Chain Theatre · 1 h | 🔵 Sonnet | ContactSheet (six slots filling on `gen_ready`, mouth 64×64 zoom), GenerationChart (SatyaCheck vs baselines from `benchmark.json` *and* the live run's p_fake per gen), CalibrationPairs (T-WA emulated vs real). Headline width bound to displayed gen. | Live chain of a sample completes on screen |
| **P9.6** | Red team · 1 h | 🔵 Sonnet | Record → words (`/redteam/words`) → WordPicker (tap to delete, drag to swap) → Make the fake → AttackPlayer (before/after) → Scan it (opens case). Lip-sync queue card with pre-made fakes. | Judge-flow done in < 45 s end to end |
| **P9.7** | Lab · 1 h | 🔵 Sonnet | ArchDiagram (static SVG from Part 4, hover shows params/latency from registry), BenchmarkWall (renders every table/plot from `/v1/benchmarks`), footer with model hashes + "reproduce: make eval". | No hard-coded metric anywhere (`grep` for digits in lab components) |
| **P9.8** | Offline tier · 45 min | 🔵 Sonnet | If `/v1/health` fails: banner + run `lib/analyze.ts` (v1) + `ort-facetrace.ts` (onnxruntime-web WASM, `public/models/facetrace-int8.onnx`) → a `mode: offline_fallback` result on the same case board (branches not run are shown as "needs the server"). | Kill the API → page still produces a labelled result |
| **P9.9** | Design critique pass · 30 min | 🔵 Sonnet | Playwright screenshots of every page at 390 / 1440 px, light/dark; review against §7.2 principles and the skill's calibration list; fix the three worst issues; remove one decorative element per page ("take one accessory off"). | Screenshots committed to `docs/screens/` |

---

## P10 — Integration, hardening, deploy, rehearsal · 3 h · owners M3 + M4

| | Sub-phase | Model | Detail | Done when |
|---|---|---|---|---|
| **P10.1** | E2E of the six wows · 1 h | 🔵 Sonnet | `tests/e2e.spec.ts` (Playwright): each §8.1 tape → expected verdict band; red-team splice → pin within ±80 ms of truth; chain completes; offline tier works. | Suite green 3 runs in a row |
| **P10.2** | Latency on CPU · 30 min | 🔵 Sonnet | `tests/test_latency_budget.py` with `CUDA_VISIBLE_DEVICES=""`, ORT 4 threads, 50 clips → p50/p95 into `benchmark.json`; tune budget guard if p95 > 8 s. | p95 recorded; A1 passes |
| **P10.3** | Deploy · 30 min | 🔵 Sonnet — **ask the user before any `vercel --prod`** | `cloudflared tunnel --url http://localhost:8000` (or named tunnel); write URL into `public/api-base.json`; add the Vercel origin to CORS; `vercel --prod --yes` from the repo root (project `fake-face-riot-paradigm-shift`); verify frozen URL loads, `/v1/health` reachable from a phone on 4G. | Frozen URL runs a live analysis end to end |
| **P10.4** | README v2 · 45 min | 🟢 Haiku (tables from JSON) + 🔵 Sonnet (prose) | Rewrite README sections that describe the system; every metric table generated from `benchmark.json` by `ml/reports/render_readme.py`; keep the demo video link; add "Reproduce" and "Limitations". Plain-text architecture diagram (no Mermaid — it did not render last time). | README renders on GitHub; numbers match the Lab page |
| **P10.5** | Rehearsal & failure drills · 1 h | human (🟢 Haiku only for checklist edits) | 3 full run-throughs of §8.3 with a timer. Drills: API down → offline tier; Wi-Fi down → localhost; a live clip misbehaves → switch to the pre-analysed case URL (keep 6 cached case IDs bookmarked); laptop on charger, sleep disabled, notifications off, browser zoom set. | Script ≤ 5:00 three times; drill checklist printed |

---

## 9.12 Finale fast-path (≈ 12 engineer-hours, 4 people in parallel)

Keep the architecture; shrink training. Order matters.

| # | Step | Who | Hours | What is cut vs. full plan |
|---|---|---|---|---|
| 1 | P0.1–P0.5 | M3 | 1.0 | — |
| 2 | P1.2 emulator (priors only) + start P1.3 phone forwarding in parallel | M3 + everyone's phones | 1.0 | Calibration fit can land later; priors first |
| 3 | P1.5 splice factory | M2 | 0.75 | No TTS insertion |
| 4 | P0.6 recordings + P1.6 MuseTalk demo tapes only (≈ 20 fakes) | M4 | 1.5 | No large fake set |
| 5 | P2.1 preprocess + P2.2 codec | M3 | 1.5 | — |
| 6 | **M2**: P3.1–P3.3 with XLS-R **frozen**, heads only, 3 epochs | M2 | 1.0 + 1 h GPU | No unfreeze, no KD |
| 7 | **M1**: P4.1–P4.3 Stage B with InfoNCE + shift (λ_kd = 0), 10 epochs on own recordings + AVLips/LipSyncTIMIT reals; Stage C on AVLips/LipSyncTIMIT fakes 2 epochs | M1 | 1.5 + 1.5 h GPU | No teacher (P2.3 skipped → ablation "no KD") |
| 8 | **M4**: SBI weights (their released EfficientNet) fine-tuned 3 epochs on gen-mixed frames | M1 | 0.5 + 0.5 h GPU | — |
| 9 | **M3**: P5.2 | M3 | 0.75 | — |
| 10 | **Nyaya**: P6.2 with temperature + conformal (skip gate MLP → logistic regression on 13-d z) | M2 | 0.75 | Simpler fusion |
| 11 | P8.1–P8.3 (skip INT8 if parity is fiddly — fp32 ORT is still < 10 s) + P8.4 splice & chain only | M3 | 2.5 | No lipsync endpoint, no dossier PDF (PNG card only) |
| 12 | P9.1–P9.3 + P9.5 + P9.6 (no 3D, no Lab atlas; Lab = BenchmarkWall only) | M4 | 4.0 | No R3F, no offline tier |
| 13 | P7.1 + P7.2 (v1 + AASIST zero-shot + pixel-only baseline) + P7.4 claims | M3 | 1.5 | Fewer baselines, no ablation retrains |
| 14 | P10.2, P10.3 (ask first), P10.5 | all | 1.5 | README v2 after judging if needed |

---

# PART 10 — CLAUDE MODEL & TOKEN STRATEGY (Max 5× plan)

## 10.1 Which model, when

| Model | Use it for | Share of sessions | Effort setting |
|---|---|---|---|
| 🔵 **Sonnet 5** (default) | Almost all implementation: data modules, preprocessing, server, endpoints, frontend components, export scripts, eval harness, README prose | ~70% | default |
| 🟣 **Opus 5.5** | Only where a subtle bug silently ruins results: **P2.3** teacher env (fairseq/CUDA), **P3.2** SpliceLoc model + losses, **P4.2** SyncFormer model + losses, **P6.2** Nyaya fusion + conformal, **P7.4** claim-guard review (10 min), and any "the metric looks too good / too bad" debugging (leakage hunts) | ~10–15% | high for P3.2/P4.2/P6.2; medium otherwise |
| 🟢 **Haiku 4.5** | Toolchain installs, downloads, running long jobs and summarising logs, generating ladders, running eval scripts, regenerating README tables from JSON, Hindi copy passes, `PROGRESS.md` updates, rote renames | ~15–20% | low |

Switch with `/model` at the start of a sub-phase; the phase tables above carry the marker.

## 10.2 Token discipline (the rules that save the most)

1. **One sub-phase per session, then `/clear`.** Start each session with the paste-in paragraph (§12.5) + the one sub-phase table row + file paths. Never "read the blueprint".
2. **Point, don't dump.** Name the exact files to open. Never ask Claude to read `data/`, `logs/`, `node_modules/`, notebooks, or whole datasets. Logs: `tail -n 30` or `grep -E "val_|EER|AUC" | tail`.
3. **Long jobs run detached.** `nohup … &` (or the harness's background mode). Don't poll every minute — check once when the ETA passes. Training is where most wasted tokens come from.
4. **Scripts over conversations.** Ask for a script that does the loop (e.g. "write `ladder.py` that processes the manifest resumably"), not for Claude to run 200 ffmpeg commands itself.
5. **Commit + `PROGRESS.md` after every sub-phase** so the next session orients from 30 lines instead of the whole chat.
6. **Opus gets only the hard file.** For P4.2, give Opus `§4.3` + the dataset batch shape + the file to write — not the frontend, not the server.
7. **Fixtures for UI work.** Record one real SSE stream to `tests/fixtures/*.jsonl`; frontend sessions build against the fixture without running the models.
8. **Subagents sparingly.** Use an Explore/Haiku subagent only for broad codebase searches; never spawn many agents for UI tweaks.
9. **Don't re-derive.** If a decision is in this document, cite it (`per D3`, `per §4.8`) instead of asking Claude to reconsider it.
10. **Screenshots over prose** for design reviews (one Playwright screenshot beats a long description).

## 10.3 Session starter template

```
Project: SatyaCheck v2 (CX0102). Blueprint: SATYACHECK_BLUEPRINT.md.
Current sub-phase: P4.2 — SyncFormer model + losses (see §4.3 and the P4.2 row of Part 9).
Relevant files: ml/train/syncformer.py (create), ml/data/dataset_sync.py (batch shape: roi [B,T,1,88,88], mel [B,4T,80], kin [B,T,8], shift [B,S], teacher_v [B,T,1024]).
Constraints: satya_core.preprocess is the only preprocessing; 8 GB VRAM; bf16; do not modify other phases' files.
Done when: overfit sanity on 32 samples passes; param count printed; unit tests in ml/tests/test_syncformer.py green. Then commit "P4.2: …" and update PROGRESS.md.
```

---

# PART 11 — RISKS, DECISIONS, OPEN QUESTIONS

## 11.1 Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Not enough time before judging | High | High | §9.12 fast-path; `[CUT]` markers; parallel owners |
| RTX 5050 (Blackwell) vs older CUDA wheels (fairseq, MuseTalk pins) | High | Medium | torch ≥ 2.7 cu128 in main env; separate envs for teacher/MuseTalk; time-boxed fallback ladders (P2.3, P1.6) |
| Gated datasets never arrive | High | Low | Plan uses open sets + own data + AVH-Align prior |
| Small labelled data → overfit | Medium | High | Real-only pre-training, frozen backbones, on-the-fly splices, identity-disjoint eval, report T-HELD |
| Leakage / shortcuts (silence, identity) | Medium | High | Silence trimming, identity splits, split test, Opus leakage review when numbers look too good |
| Emulator ≠ real WhatsApp | Medium | High | P1.3 calibration + T-WA real-forward test set shown on stage |
| INT8 drift | Medium | Low | Parity gate; fp32 fallback still in budget |
| Venue Wi-Fi / tunnel drops | High | High | Stage demo on localhost; offline tier on the frozen URL; bookmarked pre-analysed cases |
| Live clip misbehaves on stage | Medium | High | Pre-analysed case URLs; honest "that's a SUSPECT, here's why" is itself a feature |
| Recording judges — consent/privacy | Medium | Medium | Ask on camera; process locally; delete the case at the end in front of them (`DELETE` button) |
| Licences (AV-HuBERT research licence, some datasets CC BY-NC) | Low | Low | Non-commercial hackathon use; list licences in README |
| Whisper errors in Marathi/Bhojpuri | High | Low | Annotation only, never the verdict; shown as "approximate words" |
| Laptop thermals throttle during demo | Medium | Medium | Warm sessions at startup; charger; latency guard; close training jobs before demo |

## 11.2 Decision log

| # | Decision | Alternatives rejected | Reason |
|---|---|---|---|
| 1 | Real-only self-supervised pre-training + fine-tune | Supervised-only on public fakes | Generalisation to unseen (Indic) generators; small labelled data |
| 2 | Distil AV-HuBERT/AVH-Align/AASIST into small students | Deploy teachers | 10 s CPU budget |
| 3 | Our own WhatsApp emulator, calibrated | Generic JPEG/CRF augmentation; frequency-invariant architectures | arXiv 2608.28685: real codec diversity wins |
| 4 | Frame-level localisation heads | Clip-level classifiers | Fact-checker needs *where*; enables the demo |
| 5 | Mondrian conformal abstention | Fixed thresholds | Statistical guarantee per degradation level |
| 6 | Python server + browser fallback | All-in-browser | Model/tooling reality; still keeps an offline mode |
| 7 | Keep Next app at repo root | Monorepo move | Frozen Vercel URL |
| 8 | Templated explanations | LLM explanations | Hallucination risk in a misinformation tool |
| 9 | Retire neo-brutalist look for the "forensic light-table" | Keep v1 look | Instrument, not poster; frontend-design skill process |
| 10 | Keep v1 heuristics untouched as `baseline-v1` | Delete | Honest baseline row + offline fallback |

## 11.3 Open questions for the team

1. **When exactly is judging today, and how many hours can each person put in?** This picks full plan vs. §9.12.
2. **Who owns which model** (the four roles in §9.0)?
3. **May we record judges** for the red-team demo? If not, use teammates.
4. **Laptop for the stage:** confirm it's this RTX 5050 machine (the fact-checker story says CPU — we demo with GPU hidden to prove it).
5. **Keep the light-table design direction, or keep neo-brutalism?** (Part 7 is written for the former.)

---

# PART 12 — APPENDICES

## 12.1 Dependencies

**Python (main env, 3.11):**
```
torch>=2.7  torchvision  torchaudio        # cu128 index
lightning>=2.4  hydra-core>=1.3  wandb
transformers>=4.44  timm>=1.0  pytorch-grad-cam
onnx>=1.16  onnxruntime>=1.19  onnxsim  onnxscript
mediapipe>=0.10.14  opencv-python-headless  av>=12  numpy  scipy  pandas  pyarrow
soundfile  pyloudnorm  silero-vad  faster-whisper>=1.0  librosa
lightgbm>=4  scikit-learn>=1.5  shap  umap-learn
fastapi>=0.115  uvicorn[standard]  sse-starlette  sqlmodel  python-multipart  python-magic
weasyprint  pillow  qrcode  httpx  pytest
```
Separate envs: `envs/teacher` (AV-HuBERT/fairseq per its repo), `envs/musetalk` (per MuseTalk README).

**npm (add to existing):**
```
three @react-three/fiber@^9 @react-three/drei motion
d3-scale d3-shape wavesurfer.js@^7 @mediapipe/tasks-vision onnxruntime-web
zustand @tanstack/react-query zod next-intl
@radix-ui/react-tabs @radix-ui/react-tooltip @radix-ui/react-toggle-group
```
(dev) `json-schema-to-zod`, `@playwright/test`.

## 12.2 Downloads

```bash
# weights (ml/data/download/weights.sh)
W=~/.cache/satya/weights; mkdir -p $W
hf download facebook/wav2vec2-xls-r-300m config.json preprocessor_config.json pytorch_model.bin --local-dir $W/xlsr300m
hf download SpeechAntiSpoofingBenchmarks/W2V2-AASIST --local-dir $W/w2v2_aasist
wget -P $W https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt            # needed by the AASIST wrapper
wget -P $W https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/self_large_vox_433h.pt
wget -P $W https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
# AVH-Align checkpoint: checkpoints/AVH-Align_AV1M.pt — locate the official repo from the CVPR 2025 paper
#   (a mirror README is at huggingface.co/datasets/huahua123313/AVH-Alignv2)  — verify in P0.3
# SBI EfficientNet weights: github.com/mapooon/SelfBlendedImages (README links)
# MuseTalk 1.5: github.com/TMElyralab/MuseTalk (download_weights script)
# LatentSync 1.5 [CUT]: github.com/bytedance/LatentSync

# datasets (ml/data/download/datasets.sh)
hf download SherryT997/IndicTTS-Deepfake-Challenge-Data --repo-type dataset --local-dir data/raw/indictts_df
# IndicSynth: huggingface.co/datasets/vdivyasharma/IndicSynth — Hindi + Marathi configs only
# PartialSpoof: Zenodo (search "PartialSpoof database")
# AVLips: Google Drive link in github.com/AaronComo/LipFD
# LipSyncTIMIT: download link in github.com/skrantidatta/LIPINC-V2
# InDeepFake: github.com/arnabdasphd/InDeepFake
```

## 12.3 Makefile targets

```
make setup        # uv env + npm i
make models       # verify weights + registry hashes
make data         # manifests + splits
make chain        # ForwardChain ladders (bg)
make cache        # preprocess cache (bg)
make train-m2 / train-m1 / train-m4 / train-m3 / train-nyaya
make eval         # ONNX end-to-end → ml/reports/benchmark.json + plots
make export       # ONNX + INT8 + parity + registry
make serve        # uvicorn server.main:app --port 8000
make web          # next dev
make demo         # serve + web (prod build) + warmup
make tunnel       # cloudflared → writes public/api-base.json
make contracts    # codegen pydantic + zod
```

## 12.4 Glossary

| Term | Meaning |
|---|---|
| Gen-k | A clip after k WhatsApp-style forwards (k = 0 original) |
| ForwardChain | Our calibrated WhatsApp generational-loss emulator |
| RedTeam Forge | Our attack factory (splices, dubs, TTS insertions) |
| ROI | Region of interest — the 88×88 grayscale mouth crop |
| InfoNCE | Contrastive loss: pull the true audio-video pair together, push shifted pairs apart |
| KD | Knowledge distillation — a small student mimics a large teacher's features/logits |
| EER | Equal error rate — where false-accept = false-reject |
| AUC | Area under the ROC curve (0.5 chance, 1.0 perfect) |
| ECE | Expected calibration error — how far confidence is from actual accuracy |
| Conformal set | The set of labels whose non-conformity is under the calibrated threshold; guarantees coverage 1−α |
| Mondrian | Conformal calibration done separately per group (here: per generation bucket) |
| SBI | Self-Blended Images — real faces blended with themselves to create pseudo-fakes |
| BAM | Boundary-aware attention mechanism for partial-spoof localisation |

## 12.5 One paragraph, for pasting into a fresh AI session

> SatyaCheck v2 is our CX0102 (MUSA CodeX 2026) manipulated-media detector for 6–10 s vernacular WhatsApp clips. A FastAPI server (`server/`, Python 3.11, shared `satya_core/` preprocessing) runs four ONNX INT8 models in parallel on CPU — **SyncFormer** (lip-sync, self-supervised cross-modal transformer on 88×88 mouth ROIs + log-mel + lip kinematics, distilled from AV-HuBERT/AVH-Align), **SpliceLoc** (truncated XLS-R + boundary-aware frame attention for splice localisation), **FaceTrace** (EfficientNet-B0, self-blended images), **ForwardAge** (LightGBM on codec features → forward generation 0–5) — fused by **Nyaya** (gated fusion + temperature scaling + Mondrian conformal abstention → REAL/SUSPECT/FAKE with reason codes), streaming results over SSE to a Next.js 16 app kept at the repo root (frozen Vercel URL). All models train through **ForwardChain**, our WhatsApp emulator calibrated on real 5-hop forwards. Every metric shown anywhere comes from `ml/reports/benchmark.json` produced by `make eval`. The blueprint is `SATYACHECK_BLUEPRINT.md`; progress is in `PROGRESS.md`; do only the sub-phase you are given.

## 12.6 References (with links)

**Audio-visual & lip-sync detection**
- AVFF — Oorloff et al., CVPR 2024. openaccess.thecvf.com (…/Oorloff_AVFF…CVPR_2024_paper.pdf); unofficial code github.com/JoeLeelyf/OpenAVFF
- AVH-Align — Smeu et al., CVPR 2025 (unsupervised AV-HuBERT alignment; shortcut analysis)
- Investigating self-supervised representations for AV deepfake detection — CVPR 2026, arxiv.org/abs/2511.17181
- SAVe — arxiv.org/abs/2603.25140 (2026)
- A-V representation learning via audio shift prediction — Anshul, Chng, Rajan, WACV 2026
- LipFD "Lips Are Lying" — NeurIPS 2024, github.com/AaronComo/LipFD (AVLips dataset)
- LIPINC / LIPINC-V2 — Datta, Jia, Lyu; github.com/skrantidatta/LIPINC-V2 (LipSyncTIMIT)
- Lost in Translation (transcript mismatch) — Bohacek & Farid, CVPRW 2024
- X-AVDT — CVPR 2026 (AV cross-attention)
- PE-AV SyncNet — github.com/will-rice/pe-av-syncnet

**Temporal localisation**
- AV-Deepfake1M — Cai et al., ACM MM 2024, github.com/ControlNet/AV-Deepfake1M
- LAV-DF / BA-TFD+ — arxiv.org/abs/2305.01979
- UMMAFormer — arxiv.org/abs/2308.14395

**Audio anti-spoofing & partial spoof**
- SSL_Anti-spoofing (W2V2-AASIST) — Tak et al., Odyssey 2022, github.com/TakHemlata/SSL_Anti-spoofing; HF SpeechAntiSpoofingBenchmarks/W2V2-AASIST
- AASIST — Jung et al., ICASSP 2022, github.com/clovaai/aasist
- BAM — Zhong et al., Interspeech 2024, github.com/media-sec-lab/BAM
- Waveform boundary detection for partially spoofed audio — arxiv.org/abs/2211.00226
- AASIST2 short utterances — arxiv.org/abs/2309.08279
- AntiDeepfake (NII) — huggingface.co/nii-yamagishilab/xls-r-2b-anti-deepfake
- ASVspoof 5 evaluation plan/results — asvspoof.org

**Compression & social networks**
- Data Diversity, Not Frequency Invariance — arxiv.org/abs/2608.28685 (2026)
- SNVSE social-network emulation — arxiv.org/abs/2508.08765, github.com/truebees-ai/social_emulator
- PLADA — arxiv.org/abs/2506.20548
- Real-world robustness of face-swap detectors — Frontiers in AI 2026
- WhatsApp video/audio forensics theses — University of Colorado Denver NCMF (2020)
- CoFFEE H.264 codec forensics — 2024

**Indian-language data**
- InDeepFake — Pattern Recognition Letters 197 (2025), github.com/arnabdasphd/InDeepFake
- HAV-DF — arxiv.org/abs/2411.15457
- IndicTTS Deepfake Challenge — huggingface.co/datasets/SherryT997/IndicTTS-Deepfake-Challenge-Data
- IndicSynth — ACL 2025, github.com/vdivyas/IndicSynth
- MLADDC — NeurIPS 2024 D&B
- Indic-CodecFake / SATYAM — arxiv.org/abs/2604.19949

**Generators**
- MuseTalk — github.com/TMElyralab/MuseTalk · LatentSync — github.com/bytedance/LatentSync

**Calibration & abstention**
- Calibrated Selective Classification — arxiv.org/abs/2208.12084
- SCoRE — arxiv.org/abs/2603.24704
- Towards reliable deepfake detection from an uncertainty-calibration perspective — Visual Intelligence (Springer) 2025

**Runtime**
- MediaPipe Face Landmarker — developers.google.com/edge/mediapipe/solutions/vision/face_landmarker
- ONNX Runtime Web (WebGPU EP) — onnxruntime.ai/docs/tutorials/web/ep-webgpu.html
- BRAVEn / Auto-AVSR — arxiv.org/abs/2404.02098, github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages

**Design**
- Anthropic `frontend-design` skill — github.com/anthropics/claude-code/…/frontend-design/SKILL.md
