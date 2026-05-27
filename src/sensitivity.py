"""Sensitivity analysis for an unmeasured confounder.

We use a simple linear-bias model in the spirit of DoWhy's
add_unobserved_common_cause refuter, plus the Cinelli & Hazlett (2020) "robustness
value" idea adapted to AIPW.

Strategy
--------
1. Simulate an unobserved binary confounder U_i correlated with both T and Y:
       U_i | T_i = t, Y_i = y ~ Bernoulli(p_t(y))
   parameterized by two "association strengths" k_T, k_Y (log-odds).
2. Stratify the AIPW estimator by U and combine via the standard formula.
3. Sweep (k_T, k_Y) over a grid and record the resulting ATE estimate.

The output is a table (and an optional contour plot) showing how strong the
unobserved confounder would have to be to overturn the qualitative conclusion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .ate import aipw_ate
from .nuisance import cross_fit_nuisances
from .utils import ensure_dir


def simulate_unobserved_confounder(
    T: np.ndarray, Y: np.ndarray, k_T: float, k_Y: float, seed: int = 0
) -> np.ndarray:
    """Draw U ∈ {0,1} so that log-odds(U=1 | T, Y) = k_T * T + k_Y * Y."""
    rng = np.random.default_rng(seed)
    logits = k_T * T + k_Y * Y
    p = 1.0 / (1.0 + np.exp(-logits))
    return (rng.uniform(size=len(T)) < p).astype(np.int64)


def sensitivity_grid(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    k_T_grid=(0.0, 0.5, 1.0, 1.5, 2.0),
    k_Y_grid=(0.0, 0.5, 1.0, 1.5, 2.0),
    seed: int = 0,
) -> pd.DataFrame:
    """For each (k_T, k_Y), append a simulated U to X and re-estimate the ATE."""
    rows = []
    base = aipw_ate(
        X, T, Y,
        nuisances=cross_fit_nuisances(X, T, Y, seed=seed),
        seed=seed,
    ).estimate
    rows.append({"k_T": 0.0, "k_Y": 0.0, "ate": base, "note": "baseline (no U added)"})
    for kT in k_T_grid:
        for kY in k_Y_grid:
            if kT == 0.0 and kY == 0.0:
                continue
            U = simulate_unobserved_confounder(T, Y, kT, kY, seed=seed)
            X_aug = np.column_stack([X, U])
            nu = cross_fit_nuisances(X_aug, T, Y, seed=seed)
            ate = aipw_ate(X_aug, T, Y, nuisances=nu, seed=seed).estimate
            rows.append({"k_T": kT, "k_Y": kY, "ate": ate, "note": ""})
    return pd.DataFrame(rows)


def sensitivity_plot(table: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(out_path.parent)
    pivot = (
        table[table["note"] == ""]
        .pivot(index="k_Y", columns="k_T", values="ate")
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cs = ax.contourf(pivot.columns, pivot.index, pivot.values, levels=12, cmap="RdBu_r")
    contour = ax.contour(
        pivot.columns, pivot.index, pivot.values, levels=[0.0],
        colors="black", linewidths=1.5,
    )
    ax.clabel(contour, fmt="ATE = 0")
    fig.colorbar(cs, ax=ax, label="AIPW ATE with simulated U")
    ax.set_xlabel("$k_T$  (log-odds U on T)")
    ax.set_ylabel("$k_Y$  (log-odds U on Y)")
    ax.set_title("Sensitivity to a single unmeasured confounder")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
