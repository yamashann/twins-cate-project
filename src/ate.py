"""ATE estimators.

Two methods, both identified by Y(t) ⊥ T | X:

1. Outcome regression (g-formula plug-in)
       τ̂_OR = (1/n) Σ_i [μ̂_1(X_i) - μ̂_0(X_i)]

2. Augmented IPW / doubly robust
       ψ_i  = μ̂_1(X_i) - μ̂_0(X_i)
              + T_i (Y_i - μ̂_1(X_i)) / ê(X_i)
              - (1 - T_i)(Y_i - μ̂_0(X_i)) / (1 - ê(X_i))
       τ̂_AIPW = mean(ψ_i),  SE = sd(ψ_i) / sqrt(n)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .nuisance import Nuisances, cross_fit_nuisances
from .utils import normal_ci


@dataclass
class ATEResult:
    method: str
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    n: int


def outcome_regression_ate(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    nuisances: Nuisances | None = None,
    n_boot: int = 200,
    seed: int = 0,
) -> ATEResult:
    if nuisances is None:
        nuisances = cross_fit_nuisances(X, T, Y, seed=seed)
    tau_i = nuisances.mu1 - nuisances.mu0
    point = float(tau_i.mean())

    rng = np.random.default_rng(seed)
    n = len(Y)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = (nuisances.mu1[idx] - nuisances.mu0[idx]).mean()
    se = float(boots.std(ddof=1))
    lo, hi = normal_ci(point, se)
    return ATEResult("OutcomeRegression", point, se, lo, hi, n)


def aipw_ate(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    nuisances: Nuisances | None = None,
    seed: int = 0,
) -> ATEResult:
    if nuisances is None:
        nuisances = cross_fit_nuisances(X, T, Y, seed=seed)
    mu0, mu1, e = nuisances.mu0, nuisances.mu1, nuisances.e
    psi = (
        mu1 - mu0
        + T * (Y - mu1) / e
        - (1 - T) * (Y - mu0) / (1 - e)
    )
    point = float(psi.mean())
    se = float(psi.std(ddof=1) / np.sqrt(len(psi)))
    lo, hi = normal_ci(point, se)
    return ATEResult("AIPW", point, se, lo, hi, len(Y))
