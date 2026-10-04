"""Download a few short LibriSpeech clips (via librosa's example registry) into data/examples/.

These are three different read-speech speakers, ~14-17 s each, distributed by
librosa at 22.05 kHz. They are converted to 24 kHz mono WAV for the Phase 1 sweep.

Example:
    python scripts/fetch_examples.py
"""

from __future__ import annotations

import argparse
import logging

import librosa
import soundfile as sf

from codec_probing.data import load_audio
from codec_probing.utils import REPO_ROOT, resolve_path, setup_logging

logger = logging.getLogger("fetch_examples")

EXAMPLES = ("libri1", "libri2", "libri3")


def main() -> None:
    """Fetch and convert the example clips."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "examples"))
    p.add_argument("--sr", type=int, default=24000)
    args = p.parse_args()
    setup_logging()

    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in EXAMPLES:
        src = librosa.ex(key)
        wav = load_audio(src, args.sr)
        dst = out_dir / f"{key}.wav"
        sf.write(dst, wav, args.sr)
        logger.info("%s -> %s (%.2fs)", key, dst, len(wav) / args.sr)


if __name__ == "__main__":
    main()
