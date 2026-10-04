"""Thin wrapper around Kyutai's Mimi codec (Hugging Face Transformers ``MimiModel``).

API facts this module relies on (checked against transformers 5.5.4):

* ``MimiModel.encode(input_values[B, 1, T], num_quantizers=K).audio_codes`` is a
  LongTensor of shape ``[B, K, frames]`` at 12.5 frames/s.
* ``MimiModel.decode(audio_codes[B, K, frames]).audio_values`` is ``[B, 1, T']``
  for any ``1 <= K <= 32``.
* Codebook 0 is a separate *semantic* RVQ (distilled from WavLM); codebooks
  1..31 form the *acoustic* RVQ. Both quantize the same latent.
* Quantization is greedy, so the first N codes of a 32-level encode are
  identical to an N-level encode. We therefore encode once with all
  codebooks, cache that, and slice ``codes[:N]`` for each sweep point.
* Codebooks lazily build and cache their ``embed`` tensor on first use. Under
  ``torch.inference_mode()`` that cache becomes an inference tensor and later
  grad-mode calls fail, so this wrapper uses ``torch.no_grad()`` instead.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import MimiModel

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "kyutai/mimi"


@dataclass(frozen=True)
class CodecInfo:
    """Static properties of the codec needed for bitrate and length bookkeeping."""

    sample_rate: int
    frame_rate: float
    num_codebooks: int
    codebook_size: int

    @property
    def bits_per_code(self) -> float:
        """Bits needed to index one codebook entry."""
        return math.log2(self.codebook_size)

    def bitrate_bps(self, num_codebooks: int) -> float:
        """Bitrate in bits/s when transmitting the first ``num_codebooks`` codebooks."""
        return self.frame_rate * num_codebooks * self.bits_per_code


class MimiCodec:
    """Load Mimi once, then encode waveforms and decode them from the first N codebooks."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str = "cpu") -> None:
        """Load pretrained weights onto ``device`` in eval mode."""
        logger.info("Loading codec %s on %s", model_id, device)
        self.model_id = model_id
        self.device = torch.device(device)
        self.model = MimiModel.from_pretrained(model_id).to(self.device).eval()
        cfg = self.model.config
        self.info = CodecInfo(
            sample_rate=int(cfg.sampling_rate),
            frame_rate=float(cfg.frame_rate),
            num_codebooks=int(cfg.num_quantizers),
            codebook_size=int(cfg.codebook_size),
        )

    @torch.no_grad()
    def encode(self, wav: np.ndarray) -> np.ndarray:
        """Encode a mono waveform at ``info.sample_rate`` with all codebooks.

        Args:
            wav: 1-D float array.

        Returns:
            Integer codes of shape ``[num_codebooks, frames]``.
        """
        x = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).to(self.device)[None, None, :]
        codes = self.model.encode(x, num_quantizers=self.info.num_codebooks).audio_codes
        return codes[0].cpu().numpy().astype(np.int64)

    @torch.no_grad()
    def decode(self, codes: np.ndarray, num_codebooks: int | None = None, length: int | None = None) -> np.ndarray:
        """Decode audio from the first ``num_codebooks`` codebooks.

        Args:
            codes: Codes of shape ``[K, frames]`` (as returned by :meth:`encode`).
            num_codebooks: How many leading codebooks to use; ``None`` uses all of ``codes``.
            length: If set, trim or zero-pad the output to exactly this many samples.

        Returns:
            1-D float32 waveform at ``info.sample_rate``.
        """
        n = codes.shape[0] if num_codebooks is None else num_codebooks
        if not 1 <= n <= codes.shape[0]:
            raise ValueError(f"num_codebooks must be in [1, {codes.shape[0]}], got {n}")
        c = torch.from_numpy(np.ascontiguousarray(codes[:n], dtype=np.int64)).to(self.device)[None]
        wav = self.model.decode(c).audio_values[0, 0].cpu().numpy().astype(np.float32)
        if length is not None:
            if len(wav) < length:
                logger.warning("Decoded %d samples < requested %d; zero-padding", len(wav), length)
                wav = np.pad(wav, (0, length - len(wav)))
            wav = wav[:length]
        return wav

    def cache_key(self, wav: np.ndarray) -> str:
        """Content hash of (model id, codebook count, audio samples)."""
        h = hashlib.sha256()
        h.update(f"{self.model_id}|{self.info.num_codebooks}|{self.info.sample_rate}|".encode())
        h.update(np.ascontiguousarray(wav, dtype=np.float32).tobytes())
        return h.hexdigest()[:32]

    def encode_cached(self, wav: np.ndarray, cache_dir: str | Path | None) -> np.ndarray:
        """Like :meth:`encode`, but reuse codes stored on disk for identical audio.

        Codes are keyed by a hash of the audio content, so renaming or moving a
        file still hits the cache, while any change to the samples (e.g. a new
        crop length) re-encodes. Pass ``cache_dir=None`` to disable caching.
        """
        if cache_dir is None:
            return self.encode(wav)
        path = Path(cache_dir) / f"{self.cache_key(wav)}.npy"
        if path.exists():
            logger.debug("Codec cache hit: %s", path.name)
            return np.load(path).astype(np.int64)
        codes = self.encode(wav)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, codes.astype(np.int16))  # codebook_size 2048 fits in int16
        logger.debug("Codec cache write: %s", path.name)
        return codes
