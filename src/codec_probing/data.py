"""Audio loading and resampling.

Everything entering the codec is mono float32 at Mimi's 24 kHz rate.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS: frozenset[str] = frozenset({".wav", ".flac", ".ogg", ".mp3"})


def resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a mono waveform; returns the input unchanged if rates match."""
    if orig_sr == target_sr:
        return wav.astype(np.float32, copy=False)
    out = librosa.resample(wav.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr, res_type="soxr_hq")
    return out.astype(np.float32, copy=False)


def load_audio(path: str | Path, target_sr: int, max_duration_s: float | None = None) -> np.ndarray:
    """Load an audio file as mono float32 at ``target_sr``.

    Args:
        path: Audio file readable by soundfile.
        target_sr: Output sample rate.
        max_duration_s: If set, crop to this many seconds (from the start).

    Returns:
        1-D float32 array.
    """
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)  # downmix to mono
    if max_duration_s is not None:
        wav = wav[: int(round(max_duration_s * sr))]
    wav = resample(wav, sr, target_sr)
    logger.debug("Loaded %s: %.2fs @ %d Hz (orig %d Hz)", path, len(wav) / target_sr, target_sr, sr)
    return wav


def find_audio_files(inputs: Iterable[str | Path]) -> list[Path]:
    """Expand a list of files and directories into a sorted list of audio files.

    Directories are searched recursively. Missing paths raise ``FileNotFoundError``.
    """
    files: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files.extend(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXTENSIONS)
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(f"Input not found: {p}")
    return sorted(set(files))


def synthetic_speechlike(duration_s: float, sr: int, seed: int = 0) -> np.ndarray:
    """Generate a deterministic, roughly speech-like test signal.

    A harmonic source with a gliding pitch contour, shaped by a ~4 Hz syllabic
    envelope, plus a little noise. Not speech, but it has the voiced/silent
    structure that STOI and speaker embedders expect. Used by the tests.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * sr)) / sr
    f0 = 140.0 + 30.0 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    source = sum(np.sin(k * phase) / k for k in range(1, 20))
    envelope = np.clip(np.sin(2 * np.pi * 4.0 * t), 0.0, None) ** 2
    wav = envelope * source + 0.01 * rng.standard_normal(len(t))
    return (0.3 * wav / np.max(np.abs(wav))).astype(np.float32)
