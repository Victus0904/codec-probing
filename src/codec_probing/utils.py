"""Shared helpers: logging, seeding, and config loading."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def setup_logging(level: int | str = logging.INFO) -> None:
    """Configure root logging with a compact, timestamped format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str | Path, root: Path = REPO_ROOT) -> Path:
    """Resolve ``path`` against ``root`` unless it is already absolute."""
    p = Path(path)
    return p if p.is_absolute() else root / p
