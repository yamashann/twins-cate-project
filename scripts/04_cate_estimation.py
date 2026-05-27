"""CATE estimation across S, T, DR, R learners and causal forest."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cate import fit_all_cates
from src.data import load_twins
from src.nuisance import cross_fit_nuisances
from src.utils import ensure_dir


def main(seed: int = 0) -> None:
    res_dir = ensure_dir(ROOT / "results")
    data = load_twins(seed=seed)

    nu = cross_fit_nuisances(data.X, data.T, data.Y, seed=seed)
    estimates = fit_all_cates(data.X, data.T, data.Y, nuisances=nu, seed=seed)

    out = pd.DataFrame({
        "pair_id": data.pair_id,
        "ite_true": data.ite,
        "tau_S": estimates.s_learner,
        "tau_T": estimates.t_learner,
        "tau_DR": estimates.dr_learner,
        "tau_R": estimates.r_learner,
        "tau_CF": estimates.causal_forest,
        "tau_CF_lo": estimates.causal_forest_lo,
        "tau_CF_hi": estimates.causal_forest_hi,
    })
    out.to_csv(res_dir / "04_cate_predictions.csv", index=False)

    summary = pd.DataFrame({
        "method": ["S", "T", "DR", "R", "CausalForest", "Truth"],
        "mean": [
            float(estimates.s_learner.mean()),
            float(estimates.t_learner.mean()),
            float(estimates.dr_learner.mean()),
            float(estimates.r_learner.mean()),
            float(estimates.causal_forest.mean()),
            float(data.ite.mean()),
        ],
        "sd": [
            float(estimates.s_learner.std()),
            float(estimates.t_learner.std()),
            float(estimates.dr_learner.std()),
            float(estimates.r_learner.std()),
            float(estimates.causal_forest.std()),
            float(data.ite.std()),
        ],
    })
    summary.to_csv(res_dir / "04_cate_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
