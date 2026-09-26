"""Word-level timestamps via faster-whisper (small, int8, CPU). Used ONLY for the live red-team word picker and
for annotating the dossier — never for a verdict (blueprint §3.8)."""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

WEIGHTS = os.environ.get("SATYA_WEIGHTS", os.path.expanduser("~/.cache/satya/weights"))


@lru_cache(maxsize=1)
def _model():
    from faster_whisper import WhisperModel
    return WhisperModel("small", device="cpu", compute_type="int8", download_root=f"{WEIGHTS}/whisper", cpu_threads=4)


def transcribe_words(wav16k: np.ndarray, language: str | None = None) -> dict:
    """wav16k: float32 mono @16 kHz. Returns {language, words:[{w,t0,t1,p}]}."""
    segs, info = _model().transcribe(wav16k.astype(np.float32), language=language, word_timestamps=True,
                                     vad_filter=True, beam_size=1, condition_on_previous_text=False)
    words = [{"w": w.word.strip(), "t0": float(w.start), "t1": float(w.end), "p": float(w.probability)}
             for s in segs for w in (s.words or [])]
    return {"language": info.language, "language_prob": float(info.language_probability), "words": words}


def pause_segments(wav16k: np.ndarray, min_len: float = 0.22, max_len: float = 1.2) -> list[dict]:
    """ASR-free 'word-like' chunks: speech between energy valleys. Always available, always aligned to real pauses."""
    from ml.redteam.splice import energy_valleys
    dur = len(wav16k) / 16000
    cuts = [0.0] + [float(c) for c in energy_valleys(wav16k, 16000, min_gap_ms=120)] + [dur]
    segs, i = [], 0
    while i < len(cuts) - 1:
        j = i + 1
        while j < len(cuts) - 1 and cuts[j] - cuts[i] < min_len:
            j += 1
        if cuts[j] - cuts[i] >= min_len:
            segs.append({"w": f"segment {len(segs) + 1}", "t0": cuts[i], "t1": cuts[j], "p": 0.0})
        i = j
    return [s for s in segs if s["t1"] - s["t0"] <= max_len] or segs


SUPPORTED = {"hi", "mr", "en"}  # Whisper-small is unreliable elsewhere (measured: Assamese -> Telugu-script garbage)


def _degenerate(ws: list[dict], dur: float) -> bool:
    """Hallucination guard: too many zero-length tokens, or an implausible word rate."""
    if not ws:
        return True
    zero = np.mean([(w["t1"] - w["t0"]) < 0.03 for w in ws])
    rate = len(ws) / max(dur, 1e-6)
    return zero > 0.2 or rate > 6.0 or rate < 0.4


def pick_units(wav16k: np.ndarray, language: str | None = None, min_conf: float = 0.55) -> dict:
    """Word list for the picker: Whisper words when trustworthy, else pause segments. `source` tells the UI."""
    dur = len(wav16k) / 16000
    try:
        lang = language
        if lang is None:
            lang, lp, _ = _model().detect_language(wav16k.astype(np.float32))
        if lang in SUPPORTED:
            r = transcribe_words(wav16k, lang)
            ws = r["words"]
            conf = float(np.mean([w["p"] for w in ws])) if ws else 0.0
            if not _degenerate(ws, dur) and conf >= min_conf:
                return {"source": "whisper", "language": lang, "confidence": conf, "words": ws}
    except Exception:
        pass
    return {"source": "pauses", "language": None, "confidence": 0.0, "words": pause_segments(wav16k)}
