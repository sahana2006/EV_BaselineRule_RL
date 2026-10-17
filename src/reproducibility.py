"""Seed and immutable run-metadata helpers for reproducible experiments."""
from __future__ import annotations

import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> dict[str, Any]:
    """Seed all learning-side RNGs before an agent is constructed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass
    return environment_metadata(seed)


def environment_metadata(seed: int) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
