# SatyaCheck v2 — forensic checks for forwarded videos

> **Is this forwarded clip real?** SatyaCheck runs four forensic branches — lip-sync
> (SyncFormer), audio splice + synthetic voice (SpliceLoc), face blending artefacts
> (FaceTrace) and forwarding age (ForwardAge) — fuses them (Nyaya), and answers with an
> **honest verdict that can say "not sure"** instead of guessing. Even when it isn't sure,
> it essentially never calls a fake real: on unseen Wav2Lip dubs, **0–3.1 % of fakes were
> called REAL across five WhatsApp re-encode generations.**

| | |
|---|---|
| 🎯 Problem Statement | **CX0102 — Fake Face, Real Riot** |
| 👥 Team | **Paradigm Shift** · CodeX Hackathon 2026 |
| 💻 Source | **https://github.com/Abhishek222983101/paradigm-shift-codex-2026** |
| 🧪 Status | v2 — full ML pipeline + server + UI, e2e-tested |
| 📐 Architecture | `SATYACHECK_BLUEPRINT.md` (single source of truth) |
| 📊 Measured claims | `ml/reports/claims.md` — every slide sentence traces to `ml/reports/benchmark.json` |

---

## 🗄️ Backups & assets tracker (added Oct 2026)

Everything that lives outside this repo — models, recordings, archives — is mirrored
on Hugging Face under the same team accounts. **This table is the single place to find them.**

| Asset | What it is | Link |
|---|---|---|
| 🤖 **Model weights** | `facetrace.onnx`, `spliceloc.onnx`, `syncformer.int8.onnx`, `nyaya.*`, forward-age thresholds + feature spec — byte-identical to local `models/` | [huggingface.co/Abhishek1445/satyacheck-v2](https://huggingface.co/Abhishek1445/satyacheck-v2) |
| 🎙️ **Recordings & runtime data** | `data/user_recordings` (adaptation clips), `data/manifests`, `var/uploads` (everything uploaded through the web UI) — tar.gz, 272 MB | [huggingface.co/datasets/Abhishek1445/satyacheck-recordings](https://huggingface.co/datasets/Abhishek1445/satyacheck-recordings) |
| 📈 **Benchmark artifacts** | committed in this repo — `ml/reports/benchmark.json`, `ml/reports/claims.md`, plots | [`ml/reports/`](ml/reports/) |
| 🧠 **Raw teacher weights** (aasist, xlsr, whisper, wav2lip) | NOT mirrored — re-downloaded automatically from upstream repos into `~/.cache/satya/weights` on first server start (~4.6 GB) | auto |
| 🌐 **Live demo (laptop tunnel)** | cloudflared tunnel → API on :8000, web on Vercel | see `REHEARSAL.md` |

**Backup routine (Oct 2026 disk cleanup):** source lives here, models + recordings on HF.
After every demo session: `git push` this repo; if you recorded new clips, re-run the
recordings tar → `hf upload Abhishek1445/satyacheck-recordings`. HF token backup: `~/.config/hf-token-backup/`.

---

## What it does

- **Drop a clip on the Desk** (or record yourself, or pick an evidence tape). The server
  decodes it, tracks the face, and runs four branches in parallel on CPU (ONNX Runtime)
  or GPU (PyTorch).
- **Nyaya fusion** calibrates each branch by estimated forwarding generation, applies
  temperature scaling (T=0.71) and **Mondrian conformal abstention** (α=0.05, measured
  coverage 0.951). If the evidence supports both REAL and FAKE, the verdict is
  **SUSPECT — "not sure"**, never a coin flip.
- **Chain Theatre**: forward one clip through 1–5 simulated WhatsApp generations and watch
  each re-encode get analysed live. **Red team**: cut or shuffle words to forge a splice,
  then scan the result. **Lab**: every metric rendered from the evaluation harness —
  no hand-typed numbers anywhere.

## Measured results (see `ml/reports/` for all)

| Check | Result |
|---|---|
| Nyaya fused, in-domain test (n=1,728) | **AUC 0.845** vs best single branch 0.772 |
| Unseen generator (Wav2Lip, Gen 0–5) | fakes called REAL **0–3.1 %**; abstains rather than guesses |
| FaceTrace clean→extreme re-encode | 0.971 → 0.937; pixel-only baseline collapses 0.893 → 0.765 |
| SpliceLoc | AUC 0.887, boundary F1@80 ms 0.62 |
| Latency, CPU-only (GPU hidden) | **p50 4.9 s / p95 5.5 s** for a 10 s clip (gate: <8 s) |

## Run it

```bash
# 1 — API (mock mode runs with no GPU and no model weights)
. .venv/bin/activate
SATYA_MOCK=1 python -m uvicorn server.main:app --port 8000

# 2 — web
npm install && npx next dev -p 3000        # /v1/* proxies to :8000

# 3 — end to end tests
SATYA_MOCK=1 python -m pytest tests/ -q    # API + contract (14 tests)
node tests/e2e.mjs                          # browser E2E (6 scenarios)
```

Real inference (no `SATYA_MOCK`): needs the weights (`.venv`, `models/`, W&B-free) —
`SATYA_RUNTIME=ort` serves the quantised ONNX models on CPU; the default serves PyTorch on CUDA.

## Layout

```
satya_core/   runtime: preprocess (tile-fallback tracking), codec features, pipeline, verdict
ml/           training + eval harness + export (torch -> ONNX opset 18, INT8, parity gates)
server/       FastAPI: SSE streams, SQLite (24 h expiry + janitor), chains, red team, dossier
app/ lib/     Next.js 16 UI — the forensic light-table (Anek variable fonts, EN/हिं)
contracts/    AnalysisResult v2 JSON Schema — single contract for ML, server and UI
ml/reports/   benchmark.json + claims.md — the claim guard: what we may and may not say
```

## Honesty features (the point of v2)

- Conformal **abstention**: "I'm not sure" is a first-class verdict with a stated error rate.
- **Claim guard**: nothing may be claimed that isn't in `benchmark.json` from the current
  model hashes; forbidden claims are listed explicitly in `ml/reports/claims.md`.
- Every branch reports `abstained` with a reason (no face / no speech / too short) — never a guess.
- Numbers on screen come from the API; the Lab renders empty cells when a check hasn't run.

---

*Verdicts are probabilistic evidence, not proof. Clips auto-delete after 24 h.*
