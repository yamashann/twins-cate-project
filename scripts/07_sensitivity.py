"""Sensitivity analysis to a single unmeasured confounder."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_twins
from src.sensitivity import sensitivity_grid, sensitivity_plot
from src.utils import ensure_dir


def main(seed: int = 0) -> None:
    fig_dir = ensure_dir(ROOT / "figures")
    res_dir = ensure_dir(ROOT / "results")
    data = load_twins(seed=seed)

    table = sensitivity_grid(data.X, data.T, data.Y, seed=seed)
    table.to_csv(res_dir / "07_sensitivity.csv", index=False)
    print(table.to_string(index=False))
    sensitivity_plot(table, fig_dir / "07_sensitivity.png")


if __name__ == "__main__":
    main()
