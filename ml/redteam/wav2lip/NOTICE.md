# Third-party code: Wav2Lip
`wav2lip.py`, `conv.py`, `audio.py`, `hparams.py` are from https://github.com/Rudrabha/Wav2Lip (Prajwal et al., ACM MM 2020).
Upstream states the software and weights are for **academic / research / personal, non-commercial use only**.
Here it is used solely as a *held-out attack generator* to evaluate a detector (never trained on, never shipped in the product).
`audio.py` was patched for librosa >= 0.10 (keyword args) and the package import path. Weights: `wav2lip_gan.pth`
(SHA-256 ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8), loaded with `weights_only=True`.
