"""Tests for audio loading and resampling."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from codec_probing.data import find_audio_files, load_audio, resample


def test_resample_length_and_identity(wav: np.ndarray) -> None:
    assert resample(wav, 24000, 24000) is not None
    np.testing.assert_array_equal(resample(wav, 24000, 24000), wav)
    out = resample(wav, 24000, 16000)
    assert out.dtype == np.float32
    assert abs(len(out) - len(wav) * 2 // 3) <= 1


def test_load_audio_downmixes_resamples_and_crops(tmp_path, wav: np.ndarray) -> None:
    stereo = np.stack([wav, wav], axis=1)
    path = tmp_path / "stereo.wav"
    sf.write(path, stereo, 48000)  # 1 s at 48 kHz
    out = load_audio(path, target_sr=24000)
    assert out.ndim == 1 and abs(len(out) - 24000) <= 1
    cropped = load_audio(path, target_sr=24000, max_duration_s=0.5)
    assert abs(len(cropped) - 12000) <= 1


def test_find_audio_files(tmp_path, wav: np.ndarray) -> None:
    (tmp_path / "sub").mkdir()
    sf.write(tmp_path / "a.wav", wav, 24000)
    sf.write(tmp_path / "sub" / "b.flac", wav, 24000)
    (tmp_path / "notes.txt").write_text("x")
    files = find_audio_files([tmp_path])
    assert [f.name for f in files] == ["a.wav", "b.flac"]
    with pytest.raises(FileNotFoundError):
        find_audio_files([tmp_path / "missing.wav"])
