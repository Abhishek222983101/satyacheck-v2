"""In-memory AAC hop (PyAV) for on-the-fly training augmentation: splice FIRST, forward AFTER (like a real attack).
ADTS carries no encoder-delay edit list, so decoded audio is late by the codec's priming delay; we measure that lag
per (sr, ch) once and trim it so labels (splice boundaries) stay aligned."""
from __future__ import annotations
import io
from functools import lru_cache
import av, numpy as np
from scipy.signal import resample_poly

SR = 16000

def _encode_decode(x16: np.ndarray, kbps: int, sr: int, ch: int) -> np.ndarray:
    x = resample_poly(x16, sr, SR).astype(np.float32) if sr != SR else x16.astype(np.float32)
    arr = np.tile(x[None, :], (ch, 1))
    buf = io.BytesIO(); out = av.open(buf, "w", format="adts")
    st = out.add_stream("aac", rate=sr); st.bit_rate = kbps * 1000; st.layout = "mono" if ch == 1 else "stereo"
    fr = av.AudioFrame.from_ndarray(arr, format="fltp", layout=st.layout.name); fr.sample_rate = sr
    for p in st.encode(fr): out.mux(p)
    for p in st.encode(None): out.mux(p)
    out.close(); buf.seek(0)
    dec, rs, chunks = av.open(buf), av.AudioResampler(format="fltp", layout="mono", rate=SR), []
    for f in dec.decode(audio=0):
        for g in rs.resample(f): chunks.append(g.to_ndarray()[0])
    for g in rs.resample(None): chunks.append(g.to_ndarray()[0])
    dec.close()
    return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)

@lru_cache(maxsize=None)
def _lag(kbps: int, sr: int, ch: int) -> int:
    """Codec delay in 16 kHz samples, measured with a noise burst (deterministic)."""
    r = np.random.default_rng(0); probe = np.zeros(SR, np.float32); probe[4000:4000 + 1600] = r.normal(0, 0.3, 1600)
    y = _encode_decode(probe, kbps, sr, ch); n = min(len(y), len(probe))
    c = np.fft.irfft(np.fft.rfft(y[:n], 2 * n) * np.conj(np.fft.rfft(probe[:n], 2 * n)))
    lag = int(np.argmax(c[: n // 2]))
    return lag

def aac_hop(x: np.ndarray, kbps: int, sr: int, ch: int) -> np.ndarray:
    """x: float32 16 kHz mono -> same length, delay-compensated."""
    y = _encode_decode(x, kbps, sr, ch); lag = _lag(kbps, sr, ch)
    y = y[lag: lag + len(x)]
    if len(y) < len(x): y = np.pad(y, (0, len(x) - len(y)))
    return y.astype(np.float32)

def aac_chain(x: np.ndarray, gens: int, rng: np.random.Generator, priors: dict) -> np.ndarray:
    for _ in range(gens):
        kb = int(rng.choice(priors["aac_kbps"]["values"], p=priors["aac_kbps"]["p"]))
        sr = int(rng.choice(priors["audio_sr"]["values"], p=priors["audio_sr"]["p"]))
        ch = 1 if rng.random() < priors["p_mono"] else 2
        x = aac_hop(x, kb, sr, ch)
    return x
