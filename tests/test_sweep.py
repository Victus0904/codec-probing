"""End-to-end Phase 1 sweep on a 2 s synthetic wav."""

from __future__ import annotations

import csv

import numpy as np
import pytest
import soundfile as sf

from codec_probing.sweep import plot_sweep, run_sweep, write_csv

pytestmark = pytest.mark.model


def test_sweep_end_to_end(tmp_path, wav: np.ndarray, codec, embedder) -> None:
    src = tmp_path / "synthetic.wav"
    sf.write(src, wav, 24000)
    out_dir, cache_dir = tmp_path / "out", tmp_path / "cache"

    rows = run_sweep([src], codec, embedder, [1, 32], out_dir, cache_dir)
    write_csv(rows, out_dir / "results.csv")
    plot_sweep(rows, out_dir / "plot.png")

    assert [r.num_codebooks for r in rows] == [1, 32]
    assert [r.bitrate_bps for r in rows] == [137.5, 4400.0]
    for r in rows:
        assert -1.0 <= r.stoi <= 1.0 and -1.0 <= r.speaker_similarity <= 1.0
        assert sf.info(r.recon_path).frames == len(wav)
    assert (out_dir / "audio" / "synthetic_ref.wav").exists()
    assert (out_dir / "plot.png").stat().st_size > 0
    assert len(list(cache_dir.glob("*.npy"))) == 1

    with open(out_dir / "results.csv", newline="") as f:
        table = list(csv.DictReader(f))
    assert len(table) == 2
    assert set(table[0]) >= {"file", "num_codebooks", "bitrate_bps", "stoi", "speaker_similarity"}
