"""CATE estimators: S-, T-, DR-, R-learner, and causal forest.

Each estimator fits on (X, T, Y) and returns τ̂(X) by default, or τ̂(X_eval) when
an evaluation matrix is supplied. Nuisances are reused across estimators to keep
comparisons honest (same μ̂, ê for DR/R; econml-internal for causal forest).

References
----------
Künzel et al. (2019) — S-, T-, X-learner meta-learners
Nie & Wager (2021) — R-learner / quasi-oracle losses
Kennedy (2023)     — DR-learner
Wager & Athey (2018), Athey/Tibshirani/Wager (2019) — generalized random forests
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

from .nuisance import Nuisances


def _default_outcome_learner(seed: int = 0):
    return GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=seed,
    )


def _default_cate_final(seed: int = 0):
    """A more regularized regressor used as the second stage for DR/R learners,
    where the input is a noisy doubly robust pseudo-outcome and a deep GBM tends
    to overfit."""
    return GradientBoostingRegressor(
        n_estimators=100, max_depth=2, learning_rate=0.05,
        subsample=0.5, random_state=seed,
    )


def s_learner_cate(
    X: np.ndarray, T: np.ndarray, Y: np.ndarray,
    base=None, seed: int = 0, X_eval: np.ndarray | None = None,
) -> np.ndarray:
    base = base or _default_outcome_learner(seed)
    XT = np.column_stack([X, T])
    base.fit(XT, Y)
    X_pred = X if X_eval is None else X_eval
    X1 = np.column_stack([X_pred, np.ones(len(X_pred))])
    X0 = np.column_stack([X_pred, np.zeros(len(X_pred))])
    return base.predict(X1) - base.predict(X0)


def t_learner_cate(
    X: np.ndarray, T: np.ndarray, Y: np.ndarray,
    base=None, seed: int = 0, X_eval: np.ndarray | None = None,
) -> np.ndarray:
    base = base or _default_outcome_learner(seed)
    m0 = clone(base).fit(X[T == 0], Y[T == 0])
    m1 = clone(base).fit(X[T == 1], Y[T == 1])
    X_pred = X if X_eval is None else X_eval
    return m1.predict(X_pred) - m0.predict(X_pred)


def dr_learner_cate(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    nuisances: Nuisances,
    final=None,
    seed: int = 0,
    X_eval: np.ndarray | None = None,
) -> np.ndarray:
    """Kennedy (2023) DR-learner.

    pseudo_i = μ̂_1(X_i) - μ̂_0(X_i)
              + T_i (Y_i - μ̂_1(X_i)) / ê(X_i)
              - (1 - T_i)(Y_i - μ̂_0(X_i)) / (1 - ê(X_i))
    τ̂(x) = E[pseudo | X = x] fit by regressing pseudo on X.
    """
    mu0, mu1, e = nuisances.mu0, nuisances.mu1, nuisances.e
    pseudo = (
        mu1 - mu0
        + T * (Y - mu1) / e
        - (1 - T) * (Y - mu0) / (1 - e)
    )
    final = final or _default_cate_final(seed)
    final.fit(X, pseudo)
    X_pred = X if X_eval is None else X_eval
    return final.predict(X_pred)


def r_learner_cate(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    nuisances: Nuisances,
    final=None,
    seed: int = 0,
    X_eval: np.ndarray | None = None,
) -> np.ndarray:
    """Nie & Wager (2021) R-learner.

    Let m̂(x) = E[Y|X=x] = ê(x) μ̂_1(x) + (1 - ê(x)) μ̂_0(x).
    Residuals: ỹ = Y - m̂(X),  t̃ = T - ê(X).
    Minimize Σ_i (ỹ_i - t̃_i τ(X_i))^2  →  weighted regression of ỹ/t̃ on X
    with weights t̃^2.
    """
    mu0, mu1, e = nuisances.mu0, nuisances.mu1, nuisances.e
    m_hat = e * mu1 + (1.0 - e) * mu0
    y_tilde = Y - m_hat
    t_tilde = T - e
    w = t_tilde ** 2
    target = np.where(np.abs(t_tilde) > 1e-8, y_tilde / np.where(np.abs(t_tilde) > 1e-8, t_tilde, 1.0), 0.0)
    final = final or _default_cate_final(seed)
    final.fit(X, target, sample_weight=w)
    X_pred = X if X_eval is None else X_eval
    return final.predict(X_pred)


def causal_forest_cate(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    nuisances: Nuisances | None = None,
    seed: int = 0,
    X_eval: np.ndarray | None = None,
):
    """EconML CausalForestDML.

    Returns (tau_hat, lower, upper, fitted_model). `lower`/`upper` are the
    95% pointwise CIs from the honest random forest's bootstrap variance.
    The fitted model is returned so callers can extract `feature_importances_`
    or refit on a held-out split.
    """
    try:
        from econml.dml import CausalForestDML
    except ImportError as exc:  # pragma: no cover
        raise ImportError("econml is required for causal_forest_cate") from exc

    model_y = _default_outcome_learner(seed)
    from sklearn.ensemble import GradientBoostingClassifier

    model_t = GradientBoostingClassifier(random_state=seed)
    cf = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=True,
        n_estimators=500,
        min_samples_leaf=10,
        max_depth=None,
        cv=5,
        random_state=seed,
    )
    cf.fit(Y=Y, T=T, X=X)
    X_pred = X if X_eval is None else X_eval
    tau = cf.effect(X_pred).ravel()
    lower, upper = cf.effect_interval(X_pred, alpha=0.05)
    return tau, np.asarray(lower).ravel(), np.asarray(upper).ravel(), cf


@dataclass
class CATEEstimates:
    s_learner: np.ndarray
    t_learner: np.ndarray
    dr_learner: np.ndarray
    r_learner: np.ndarray
    causal_forest: np.ndarray
    causal_forest_lo: np.ndarray
    causal_forest_hi: np.ndarray
    causal_forest_model: object = None


def fit_all_cates(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    nuisances: Nuisances,
    seed: int = 0,
) -> CATEEstimates:
    s = s_learner_cate(X, T, Y, seed=seed)
    t = t_learner_cate(X, T, Y, seed=seed)
    dr = dr_learner_cate(X, T, Y, nuisances=nuisances, seed=seed)
    r = r_learner_cate(X, T, Y, nuisances=nuisances, seed=seed)
    cf_tau, cf_lo, cf_hi, cf_model = causal_forest_cate(X, T, Y, nuisances=nuisances, seed=seed)
    return CATEEstimates(
        s_learner=s, t_learner=t, dr_learner=dr, r_learner=r,
        causal_forest=cf_tau, causal_forest_lo=cf_lo, causal_forest_hi=cf_hi,
        causal_forest_model=cf_model,
    )
