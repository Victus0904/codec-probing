"""Shared fixtures: 2 s of synthetic audio and session-scoped pretrained models."""

from __future__ import annotations

import numpy as np
import pytest

from codec_probing.data import synthetic_speechlike

SR = 24000
DURATION_S = 2.0


@pytest.fixture(scope="session")
def wav() -> np.ndarray:
    """2 s of deterministic speech-like audio at 24 kHz."""
    return synthetic_speechlike(DURATION_S, SR, seed=0)


@pytest.fixture(scope="session")
def codec():
    """Mimi on CPU (weights downloaded on first run, then read from the HF cache)."""
    from codec_probing.codec import MimiCodec

    return MimiCodec(device="cpu")


@pytest.fixture(scope="session")
def embedder(tmp_path_factory):
    """ECAPA speaker embedder on CPU."""
    from codec_probing.metrics import SpeakerEmbedder
    from codec_probing.utils import REPO_ROOT

    return SpeakerEmbedder(savedir=REPO_ROOT / "outputs" / "pretrained" / "ecapa", device="cpu")
