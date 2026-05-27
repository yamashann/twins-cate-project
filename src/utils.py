"""Small shared utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w") as f:
        json.dump(_jsonify(obj), f, indent=2, sort_keys=True)


def _jsonify(o):
    if isinstance(o, dict):
        return {str(k): _jsonify(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonify(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def normal_ci(mean: float, se: float, alpha: float = 0.05) -> tuple[float, float]:
    from scipy.stats import norm

    z = norm.ppf(1.0 - alpha / 2.0)
    return mean - z * se, mean + z * se
