"""ATE refutation tests (DoWhy-style, implemented locally for transparency).

For each test we re-estimate the ATE under a perturbation that should leave the
true effect unchanged (random covariate, subsample) or send it to zero (placebo
treatment), and compare to the original estimate. The output is a small table that
can be dropped straight into the report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .ate import aipw_ate
from .nuisance import cross_fit_nuisances


@dataclass
class RefutationResult:
    test: str
    original: float
    refuted: float
    delta: float
    pvalue: float | None = None


Estimator = Callable[[np.ndarray, np.ndarray, np.ndarray], float]


def _default_estimator(X, T, Y, seed: int = 0) -> float:
    nu = cross_fit_nuisances(X, T, Y, seed=seed)
    return aipw_ate(X, T, Y, nuisances=nu, seed=seed).estimate


def placebo_treatment(
    X: np.ndarray, T: np.ndarray, Y: np.ndarray,
    n_iter: int = 50, seed: int = 0, estimator: Estimator | None = None,
) -> RefutationResult:
    """Permute T uniformly at random and re-estimate. Should be ≈ 0."""
    estimator = estimator or _default_estimator
    original = estimator(X, T, Y)
    rng = np.random.default_rng(seed)
    refuted = np.empty(n_iter)
    for i in range(n_iter):
        T_perm = rng.permutation(T)
        refuted[i] = estimator(X, T_perm, Y)
    refuted_mean = float(refuted.mean())
    # Two-sided p for H0: refuted == 0.
    se = float(refuted.std(ddof=1) / np.sqrt(n_iter))
    from scipy.stats import norm

    p = float(2 * (1 - norm.cdf(abs(refuted_mean / se))))
    return RefutationResult("placebo_treatment", original, refuted_mean, refuted_mean - 0.0, p)


def random_common_cause(
    X: np.ndarray, T: np.ndarray, Y: np.ndarray,
    n_iter: int = 20, seed: int = 0, estimator: Estimator | None = None,
) -> RefutationResult:
    """Add an irrelevant N(0,1) covariate and re-estimate. Should ≈ original."""
    estimator = estimator or _default_estimator
    original = estimator(X, T, Y)
    rng = np.random.default_rng(seed)
    refuted = np.empty(n_iter)
    for i in range(n_iter):
        Z = rng.normal(size=(X.shape[0], 1))
        refuted[i] = estimator(np.column_stack([X, Z]), T, Y)
    refuted_mean = float(refuted.mean())
    return RefutationResult(
        "random_common_cause", original, refuted_mean, refuted_mean - original,
    )


def subset_refuter(
    X: np.ndarray, T: np.ndarray, Y: np.ndarray,
    fraction: float = 0.8, n_iter: int = 30, seed: int = 0,
    estimator: Estimator | None = None,
) -> RefutationResult:
    """Re-estimate on random subsamples; should be stable up to sampling noise."""
    estimator = estimator or _default_estimator
    original = estimator(X, T, Y)
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = int(round(fraction * n))
    refuted = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.choice(n, size=k, replace=False)
        refuted[i] = estimator(X[idx], T[idx], Y[idx])
    refuted_mean = float(refuted.mean())
    return RefutationResult(
        "subset_refuter", original, refuted_mean, refuted_mean - original,
    )


def run_all_refutations(
    X: np.ndarray, T: np.ndarray, Y: np.ndarray,
    seed: int = 0,
) -> list[RefutationResult]:
    return [
        placebo_treatment(X, T, Y, seed=seed),
        random_common_cause(X, T, Y, seed=seed),
        subset_refuter(X, T, Y, seed=seed),
    ]
