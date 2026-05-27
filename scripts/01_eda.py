"""Run exploratory analysis and overlap diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_twins
from src.eda import (
    estimate_propensity,
    mortality_by_treatment_plot,
    overlap_plot,
    smd_plot,
    standardized_mean_differences,
    summary_table,
)
from src.utils import ensure_dir


def main(seed: int = 0) -> None:
    fig_dir = ensure_dir(ROOT / "figures")
    res_dir = ensure_dir(ROOT / "results")

    data = load_twins(seed=seed)
    summary = summary_table(data)
    summary.to_csv(res_dir / "01_summary.csv", index=False)
    print(summary.to_string(index=False))

    smd = standardized_mean_differences(data)
    smd.to_csv(res_dir / "01_smd.csv", index=False)
    smd_plot(smd, fig_dir / "01_smd_top20.png")

    e_hat = estimate_propensity(data, seed=seed)
    overlap_plot(e_hat, data.T, fig_dir / "01_overlap.png")
    mortality_by_treatment_plot(data, fig_dir / "01_mortality_by_T.png")

    print(f"\nSaved EDA artifacts to {fig_dir} and {res_dir}")


if __name__ == "__main__":
    main()
