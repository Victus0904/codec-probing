"""Tests for STOI and speaker similarity."""

from __future__ import annotations

import numpy as np
import pytest

from codec_probing.metrics import cosine_similarity, stoi

SR = 24000


def test_cosine_similarity() -> None:
    a = np.array([1.0, 0.0])
    assert cosine_similarity(a, a) == pytest.approx(1.0)
    assert cosine_similarity(a, np.array([0.0, 2.0])) == pytest.approx(0.0)
    assert cosine_similarity(a, -a) == pytest.approx(-1.0)


def test_stoi_identity_and_noise(wav: np.ndarray) -> None:
    assert stoi(wav, wav, SR) == pytest.approx(1.0, abs=1e-6)
    noise = np.random.default_rng(0).standard_normal(len(wav)).astype(np.float32) * 0.3
    # Synthetic harmonics make the noise floor higher than for real speech (~0.4 here).
    assert stoi(wav, noise, SR) < 0.6
    # Degrades monotonically with more additive noise.
    mild, heavy = wav + 0.1 * noise, wav + 1.0 * noise
    assert stoi(wav, mild, SR) > stoi(wav, heavy, SR)


def test_stoi_handles_length_mismatch(wav: np.ndarray) -> None:
    assert stoi(wav, wav[:-100], SR) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.model
def test_speaker_embedder(embedder, wav: np.ndarray) -> None:
    emb = embedder.embed(wav, SR)
    assert emb.shape == (192,)
    assert embedder.similarity(wav, wav, SR) == pytest.approx(1.0, abs=1e-5)
    noise = np.random.default_rng(1).standard_normal(len(wav)).astype(np.float32) * 0.1
    assert embedder.similarity(wav, noise, SR) < 0.9
