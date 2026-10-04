"""Reconstruction metrics: intelligibility (STOI) and speaker similarity (ECAPA)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from pystoi import stoi as _pystoi

from codec_probing.data import resample

logger = logging.getLogger(__name__)

DEFAULT_ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"


def _align(ref: np.ndarray, deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trim two signals to their common length."""
    n = min(len(ref), len(deg))
    return ref[:n], deg[:n]


def stoi(ref: np.ndarray, deg: np.ndarray, sr: int, extended: bool = False) -> float:
    """Short-Time Objective Intelligibility of ``deg`` against clean ``ref``.

    Roughly in [0, 1]; higher is more intelligible. pystoi resamples to 10 kHz
    internally and drops silent frames, so only the speech band is scored.
    """
    ref, deg = _align(ref, deg)
    return float(_pystoi(ref.astype(np.float64), deg.astype(np.float64), sr, extended=extended))


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """Cosine similarity between two 1-D vectors."""
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), eps))


class SpeakerEmbedder:
    """SpeechBrain ECAPA-TDNN speaker embeddings (192-d, trained on VoxCeleb at 16 kHz)."""

    def __init__(
        self,
        source: str = DEFAULT_ECAPA_SOURCE,
        savedir: str | Path = "outputs/pretrained/ecapa",
        sample_rate: int = 16000,
        device: str = "cpu",
    ) -> None:
        """Download (once) and load the ECAPA model."""
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        logger.info("Loading speaker model %s on %s", source, device)
        self.sample_rate = sample_rate
        # COPY instead of the default symlink strategy: symlinks need admin or
        # Developer Mode on Windows.
        self.model = EncoderClassifier.from_hparams(
            source=source,
            savedir=str(savedir),
            run_opts={"device": device},
            local_strategy=LocalStrategy.COPY,
        )
        self.model.eval()

    @torch.no_grad()
    def embed(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """Return a 192-d speaker embedding for a mono waveform at ``sr``."""
        x = torch.from_numpy(resample(wav, sr, self.sample_rate))[None]
        return self.model.encode_batch(x)[0, 0].cpu().numpy()

    def similarity(self, ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
        """Cosine similarity of the speaker embeddings of two waveforms."""
        return cosine_similarity(self.embed(ref, sr), self.embed(deg, sr))
