"""Phase 1 entry point: Mimi reconstruction quality vs. number of codebooks.

Example:
    python scripts/run_sweep.py --config configs/phase1.yaml
    python scripts/run_sweep.py --inputs my.wav other_dir/ --num-codebooks 1 8 32
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from codec_probing.codec import MimiCodec
from codec_probing.data import find_audio_files
from codec_probing.metrics import SpeakerEmbedder
from codec_probing.sweep import plot_sweep, run_sweep, write_csv
from codec_probing.utils import REPO_ROOT, load_config, resolve_path, set_seed, setup_logging

logger = logging.getLogger("run_sweep")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments; any value given here overrides the YAML config."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "phase1.yaml"), help="YAML config path")
    p.add_argument("--inputs", nargs="+", help="audio files and/or directories")
    p.add_argument("--num-codebooks", nargs="+", type=int, help="N values to sweep")
    p.add_argument("--output-dir", help="where to write audio, CSV, and plot")
    p.add_argument("--max-duration", type=float, help="crop inputs to this many seconds")
    p.add_argument("--device", help="cpu or cuda")
    p.add_argument("--no-cache", action="store_true", help="always re-encode (ignore the code cache)")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the sweep described by the config plus CLI overrides."""
    args = parse_args(argv)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 0)))

    device = args.device or cfg.get("device", "cpu")
    inputs = args.inputs or [str(resolve_path(p)) for p in cfg["data"]["inputs"]]
    ns = sorted(args.num_codebooks or cfg["sweep"]["num_codebooks"])
    max_dur = args.max_duration if args.max_duration is not None else cfg["data"].get("max_duration_s")
    out_dir = resolve_path(args.output_dir or cfg["output"]["dir"])
    cache_dir = None if args.no_cache else resolve_path(cfg["output"]["cache_dir"])

    files = find_audio_files(inputs)
    if not files:
        raise SystemExit(f"No audio files found in {inputs}")
    logger.info("%d input file(s); N = %s; output -> %s", len(files), ns, out_dir)

    codec = MimiCodec(cfg["codec"]["model_id"], device=device)
    if codec.info.sample_rate != cfg["codec"]["sample_rate"]:
        logger.warning("Config sample_rate %s != model's %s; using the model's",
                       cfg["codec"]["sample_rate"], codec.info.sample_rate)
    bad = [n for n in ns if not 1 <= n <= codec.info.num_codebooks]
    if bad:
        raise SystemExit(f"num_codebooks must be in [1, {codec.info.num_codebooks}], got {bad}")

    spk = cfg["speaker_model"]
    embedder = SpeakerEmbedder(spk["source"], resolve_path(spk["savedir"]), spk["sample_rate"], device)

    rows = run_sweep(files, codec, embedder, ns, out_dir, cache_dir, max_dur)
    write_csv(rows, out_dir / "results.csv")
    plot_sweep(rows, out_dir / "stoi_speaker_vs_n.png")


if __name__ == "__main__":
    main()
