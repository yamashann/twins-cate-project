"""ATE refutation tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_twins
from src.refutation import run_all_refutations
from src.utils import ensure_dir


def main(seed: int = 0) -> None:
    res_dir = ensure_dir(ROOT / "results")
    data = load_twins(seed=seed)
    results = run_all_refutations(data.X, data.T, data.Y, seed=seed)
    df = pd.DataFrame([r.__dict__ for r in results])
    df.to_csv(res_dir / "05_refutations.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
