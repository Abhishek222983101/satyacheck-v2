import numpy as np, pytest
from ml.redteam.splice import splice_spans, random_splice, energy_valleys, FRAME_S

SR = 16000

def speechlike(seed=0, dur=8.0):
    """Syllable bursts (harmonic tone x envelope) separated by 80-200 ms pauses; deterministic."""
    rng = np.random.default_rng(seed); out = []; t = 0.0
    while t < dur:
        L = rng.uniform(0.15, 0.45); f0 = rng.uniform(110, 220); n = int(L * SR)
        tt = np.arange(n) / SR
        env = np.hanning(n)
        seg = sum(np.sin(2 * np.pi * f0 * k * tt) / k for k in (1, 2, 3, 4)) * env * 0.3
        out.append(seg); gap = rng.uniform(0.08, 0.2); out.append(rng.normal(0, 1e-3, int(gap * SR))); t += L + gap
    return np.concatenate(out)[: int(dur * SR)].astype(np.float32)

def test_delete_duration_and_boundary():
    x = speechlike(); span = (2.0, 2.5)
    r = splice_spans(x, SR, "delete", span, xf_ms=8)
    assert abs(r.duration - (len(x) / SR - 0.5)) < 0.03           # ~0.5 s shorter (snap + crossfade slack)
    assert len(r.boundaries) == 1 and abs(r.boundaries[0] - 2.0) < 0.02   # join sits at the old span start
    assert r.frame_boundary[int(r.boundaries[0] / FRAME_S)] == 1
    assert r.frame_spoof.sum() == 0                                # deletion leaves no foreign material
    assert r.removed[0][0] == pytest.approx(2.0, abs=0.01)

def test_join_is_continuous_no_click():
    x = speechlike(1); r = splice_spans(x, SR, "delete", (3.0, 3.4), xf_ms=10)
    k = int(r.boundaries[0] * SR)
    step = np.abs(np.diff(r.audio[k - 5:k + 5])).max()
    assert step < 0.35                                             # no full-scale discontinuity

def test_duplicate_adds_one_span_and_two_boundaries():
    x = speechlike(2); r = splice_spans(x, SR, "duplicate", (2.0, 2.4))
    assert len(r.boundaries) == 2 and len(r.spoof_spans) == 1
    assert abs(r.duration - (len(x) / SR + 0.4)) < 0.05
    t0, t1 = r.spoof_spans[0]; assert 0.35 < t1 - t0 < 0.45
    assert r.frame_spoof[int((t0 + t1) / 2 / FRAME_S)] == 1

def test_swap_preserves_length_three_boundaries():
    x = speechlike(3); r = splice_spans(x, SR, "swap", (1.0, 1.4), (2.0, 2.4))
    assert abs(r.duration - len(x) / SR) < 0.05 and len(r.boundaries) == 4
    assert len(r.spoof_spans) == 2

def test_insert_matches_level():
    x = speechlike(4); other = speechlike(5)[: SR // 2] * 0.1
    r = splice_spans(x, SR, "insert", (2.0, 2.01), other=other)
    t0, t1 = r.spoof_spans[0]
    inserted = r.audio[int(t0 * SR): int(t1 * SR)]
    ctx = np.concatenate([r.audio[int((t0 - 0.5) * SR): int(t0 * SR)], r.audio[int(t1 * SR): int((t1 + 0.5) * SR)]])
    ratio = np.sqrt(np.mean(inserted ** 2)) / np.sqrt(np.mean(ctx ** 2)); assert 0.3 < ratio < 3

def test_bad_span_raises():
    with pytest.raises(ValueError): splice_spans(speechlike(), SR, "delete", (9.0, 9.5))

def test_random_splice_labels_consistent_and_deterministic():
    x = speechlike(6)
    a = random_splice(x, SR, np.random.default_rng(5), op="delete"); b = random_splice(x, SR, np.random.default_rng(5), op="delete")
    assert a is not None and np.array_equal(a.audio, b.audio) and a.boundaries == b.boundaries
    assert a.frame_boundary.sum() >= 3 and len(a.frame_boundary) == int(np.ceil(len(a.audio) / SR / FRAME_S))
    for op in ("swap", "duplicate"):
        assert random_splice(x, SR, np.random.default_rng(1), op=op) is not None

def test_valleys_are_in_pauses():
    x = speechlike(7); v = energy_valleys(x, SR)
    assert len(v) > 5
    e = np.sqrt(np.convolve(x ** 2, np.ones(160) / 160, mode="same"))
    assert np.median([e[int(t * SR)] for t in v]) < np.median(e[e > 0]) * 0.5


def test_replace_span_preserves_length_and_labels():
    from ml.redteam.splice import replace_span
    x = speechlike(8); other = speechlike(9)
    r = replace_span(x, SR, (2.0, 2.5), other=other)
    assert len(r.audio) == len(x)                                   # exact same length: audio still lines up with video
    assert len(r.boundaries) == 2 and 0.4 < r.boundaries[1] - r.boundaries[0] < 0.6
    assert len(r.spoof_spans) == 1 and r.frame_spoof[int(2.25 / FRAME_S)] == 1
    r2 = replace_span(x, SR, (1.0, 1.4), src_start=5.0)             # from elsewhere in the same clip
    assert len(r2.audio) == len(x) and not np.array_equal(r2.audio, x)
    with pytest.raises(ValueError): replace_span(x, SR, (9.0, 9.9), other=other)
