"""Tests for the Mimi wrapper on 2 s of synthetic audio."""

from __future__ import annotations

import numpy as np
import pytest

from codec_probing.codec import CodecInfo

pytestmark = pytest.mark.model


def test_bitrate_formula() -> None:
    info = CodecInfo(sample_rate=24000, frame_rate=12.5, num_codebooks=32, codebook_size=2048)
    assert info.bits_per_code == 11
    assert info.bitrate_bps(1) == 137.5
    assert info.bitrate_bps(8) == 1100.0  # Mimi's published 1.1 kbps operating point
    assert info.bitrate_bps(32) == 4400.0


def test_codec_info_matches_mimi(codec) -> None:
    assert codec.info == CodecInfo(sample_rate=24000, frame_rate=12.5, num_codebooks=32, codebook_size=2048)


def test_encode_shape_and_range(codec, wav: np.ndarray) -> None:
    codes = codec.encode(wav)
    assert codes.shape == (32, 25)  # 2 s x 12.5 Hz
    assert codes.min() >= 0 and codes.max() < 2048


def test_prefix_slicing_equals_direct_encode(codec, wav: np.ndarray) -> None:
    """Sweep relies on this: first N of a 32-level encode == an N-level encode."""
    import torch

    codes = codec.encode(wav)
    x = torch.from_numpy(wav)[None, None]
    for n in (1, 2, 8):
        direct = codec.model.encode(x, num_quantizers=n).audio_codes[0].numpy()
        np.testing.assert_array_equal(codes[:n], direct)


@pytest.mark.parametrize("n", [1, 32])
def test_decode_length(codec, wav: np.ndarray, n: int) -> None:
    out = codec.decode(codec.encode(wav), num_codebooks=n, length=len(wav))
    assert out.shape == wav.shape and out.dtype == np.float32
    assert np.isfinite(out).all()


def test_decode_rejects_bad_n(codec, wav: np.ndarray) -> None:
    codes = codec.encode(wav)
    for bad in (0, 33):
        with pytest.raises(ValueError):
            codec.decode(codes, num_codebooks=bad)


def test_encode_cached_hits_disk(codec, wav: np.ndarray, tmp_path, monkeypatch) -> None:
    first = codec.encode_cached(wav, tmp_path)
    assert len(list(tmp_path.glob("*.npy"))) == 1

    def fail(*_a, **_k):
        raise AssertionError("encode() called despite a cache hit")

    monkeypatch.setattr(codec, "encode", fail)
    np.testing.assert_array_equal(codec.encode_cached(wav, tmp_path), first)
    # Different audio -> different key -> would re-encode.
    assert codec.cache_key(wav) != codec.cache_key(wav * 0.5)
