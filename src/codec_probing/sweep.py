"""Phase 1: reconstruct audio from the first N Mimi codebooks and score it."""

from __future__ import annotations

import csv
import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from codec_probing.codec import MimiCodec  # noqa: E402
from codec_probing.data import load_audio  # noqa: E402
from codec_probing.metrics import SpeakerEmbedder, cosine_similarity, stoi  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class SweepRow:
    """One (input file, N codebooks) result."""

    file: str
    duration_s: float
    num_codebooks: int
    bitrate_bps: float
    stoi: float
    speaker_similarity: float
    decode_time_s: float
    recon_path: str


def run_sweep(
    files: Sequence[Path],
    codec: MimiCodec,
    embedder: SpeakerEmbedder,
    num_codebooks: Sequence[int],
    out_dir: Path,
    cache_dir: Path | None,
    max_duration_s: float | None = None,
) -> list[SweepRow]:
    """Encode each file once, decode with each N, and compute metrics.

    Writes the 24 kHz reference and each reconstruction to ``out_dir/audio/``.
    """
    sr = codec.info.sample_rate
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows: list[SweepRow] = []

    for i, path in enumerate(files, 1):
        wav = load_audio(path, sr, max_duration_s)
        duration = len(wav) / sr
        logger.info("[%d/%d] %s (%.2fs)", i, len(files), path.name, duration)
        sf.write(audio_dir / f"{path.stem}_ref.wav", wav, sr)

        t0 = time.perf_counter()
        codes = codec.encode_cached(wav, cache_dir)
        logger.info("  codes %s in %.2fs", tuple(codes.shape), time.perf_counter() - t0)
        ref_emb = embedder.embed(wav, sr)

        for n in num_codebooks:
            t0 = time.perf_counter()
            recon = codec.decode(codes, num_codebooks=n, length=len(wav))
            dt = time.perf_counter() - t0
            recon_path = audio_dir / f"{path.stem}_n{n:02d}.wav"
            sf.write(recon_path, recon, sr)
            row = SweepRow(
                file=path.name,
                duration_s=round(duration, 3),
                num_codebooks=n,
                bitrate_bps=codec.info.bitrate_bps(n),
                stoi=stoi(wav, recon, sr),
                speaker_similarity=cosine_similarity(ref_emb, embedder.embed(recon, sr)),
                decode_time_s=round(dt, 4),
                recon_path=str(recon_path),
            )
            logger.info(
                "  N=%2d  %6.0f bps  STOI=%.3f  spk_sim=%.3f  decode=%.2fs",
                n, row.bitrate_bps, row.stoi, row.speaker_similarity, dt,
            )
            rows.append(row)
    return rows


def write_csv(rows: Sequence[SweepRow], path: Path) -> None:
    """Write sweep results to CSV (one row per file x N)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(SweepRow.__dataclass_fields__))
        writer.writeheader()
        for r in rows:
            d = asdict(r)
            d["stoi"] = f"{r.stoi:.4f}"
            d["speaker_similarity"] = f"{r.speaker_similarity:.4f}"
            writer.writerow(d)
    logger.info("Wrote %s", path)


def plot_sweep(rows: Sequence[SweepRow], path: Path) -> None:
    """Plot STOI and speaker similarity vs. N as two stacked panels sharing the x-axis.

    Thin grey lines are individual files; the blue line is the mean across files.
    The top axis labels each N with its bitrate.
    """
    ns = sorted({r.num_codebooks for r in rows})
    files = sorted({r.file for r in rows})
    bitrate = {r.num_codebooks: r.bitrate_bps for r in rows}
    ink, muted, grid, accent = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"

    fig, axes = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True, constrained_layout=True)
    panels = [("stoi", "STOI (intelligibility)"), ("speaker_similarity", "Speaker similarity (ECAPA cosine)")]
    for ax, (key, label) in zip(axes, panels):
        per_file = np.array(
            [[next(getattr(r, key) for r in rows if r.file == f and r.num_codebooks == n) for n in ns] for f in files]
        )
        if len(files) > 1:
            for y in per_file:
                ax.plot(ns, y, color="#b5b4ae", linewidth=1, zorder=1)
        mean = per_file.mean(axis=0)
        ax.plot(ns, mean, color=accent, linewidth=2, marker="o", markersize=6, zorder=3,
                markeredgecolor="white", markeredgewidth=1.5)
        for n, y in zip(ns, mean):
            ax.annotate(f"{y:.2f}", (n, y), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, color=muted)
        ax.set_ylabel(label, color=ink)
        ax.set_ylim(min(0.0, float(per_file.min()) - 0.05), 1.05)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(muted)
        ax.spines["bottom"].set_color(muted)
        ax.tick_params(colors=muted)

    ax = axes[-1]
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.minorticks_off()
    ax.set_xlabel("Number of codebooks N (first N RVQ levels)", color=ink)

    top = axes[0].secondary_xaxis("top")
    top.set_xticks(ns, [f"{bitrate[n] / 1000:.2g}k" for n in ns])
    top.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    top.tick_params(colors=muted, labelsize=8)
    top.set_xlabel("bitrate (bits/s)", color=muted, fontsize=9)
    top.spines["top"].set_visible(False)

    note = f"mean of {len(files)} files (grey: individual files)" if len(files) > 1 else files[0]
    fig.suptitle(f"Mimi reconstruction vs. number of codebooks\n{note}", fontsize=11, color=ink)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", path)
