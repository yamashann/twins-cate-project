"""CATE validation: twin-pair benchmark, R-loss, DR-score MSE, calibration, RATE,
GATEs, best linear projection, and CATE-adapted refutations.

For the Twins dataset both potential outcomes are observed for every pair, so the
per-pair true ITE = Y(1) - Y(0) is available and is the strongest validation signal.
The remaining diagnostics are the standard model-selection / heterogeneity-check
toolkit that does not assume access to ground truth and is what we would use on a
real observational study.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
import statsmodels.api as sm

from .nuisance import Nuisances


# ---- Ground-truth comparisons (Twins-specific) ------------------------------

@dataclass
class GroundTruthScore:
    method: str
    pehe: float          # sqrt mean squared error vs. true ITE
    bias: float          # mean(tau_hat - ite)
    spearman: float
    kendall: float


def pehe(tau_hat: np.ndarray, ite: np.ndarray) -> float:
    return float(np.sqrt(np.mean((tau_hat - ite) ** 2)))


def ground_truth_scores(
    methods: dict[str, np.ndarray], ite: np.ndarray
) -> pd.DataFrame:
    rows = []
    for name, tau in methods.items():
        rows.append(
            GroundTruthScore(
                method=name,
                pehe=pehe(tau, ite),
                bias=float((tau - ite).mean()),
                spearman=float(spearmanr(tau, ite).statistic),
                kendall=float(kendalltau(tau, ite).statistic),
            )
        )
    return pd.DataFrame([r.__dict__ for r in rows])


def calibration_table(
    tau_hat: np.ndarray, ite: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Bin pairs by predicted CATE; report bin-mean(predicted) vs bin-mean(true)."""
    order = np.argsort(tau_hat)
    bins = np.array_split(order, n_bins)
    rows = []
    for k, idx in enumerate(bins):
        rows.append(
            {
                "bin": k + 1,
                "n": len(idx),
                "mean_predicted": float(tau_hat[idx].mean()),
                "mean_true_ite": float(ite[idx].mean()),
            }
        )
    return pd.DataFrame(rows)


# ---- Ground-truth-free diagnostics ------------------------------------------

def r_loss(
    tau_hat: np.ndarray, Y: np.ndarray, T: np.ndarray, nuisances: Nuisances
) -> float:
    """Nie & Wager (2021) R-loss.

    L_R(τ) = mean[ ((Y - m̂(X)) - (T - ê(X)) · τ(X))^2 ]
    """
    mu0, mu1, e = nuisances.mu0, nuisances.mu1, nuisances.e
    m_hat = e * mu1 + (1 - e) * mu0
    y_tilde = Y - m_hat
    t_tilde = T - e
    return float(np.mean((y_tilde - t_tilde * tau_hat) ** 2))


def dr_score_mse(
    tau_hat: np.ndarray, Y: np.ndarray, T: np.ndarray, nuisances: Nuisances
) -> float:
    """MSE between τ̂(X) and the DR pseudo-outcome.

    pseudo_i is unbiased for τ(X_i) in expectation, so its MSE vs τ̂ is a valid
    target for model selection (Alaa & van der Schaar, 2019; Saito & Yasui, 2020).
    """
    mu0, mu1, e = nuisances.mu0, nuisances.mu1, nuisances.e
    pseudo = (
        mu1 - mu0
        + T * (Y - mu1) / e
        - (1 - T) * (Y - mu0) / (1 - e)
    )
    return float(np.mean((tau_hat - pseudo) ** 2))


def heldout_cate_predictions(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_splits: int = 5,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Out-of-fold CATE predictions for model-selection diagnostics.

    Each fold fits the CATE learner on the training folds and predicts τ̂(x) for
    the held-out fold. The resulting predictions let R-loss and DR-score MSE
    evaluate the CATE stage out of sample, instead of rewarding learners for
    fitting their own pseudo-outcomes in sample.
    """
    from .cate import (
        causal_forest_cate,
        dr_learner_cate,
        r_learner_cate,
        s_learner_cate,
        t_learner_cate,
    )
    from .nuisance import cross_fit_nuisances

    methods = ["S", "T", "DR", "R", "CausalForest"]
    tau_oof = {name: np.full(len(Y), np.nan) for name in methods}
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, T)):
        fold_seed = seed + fold
        X_tr, T_tr, Y_tr = X[train_idx], T[train_idx], Y[train_idx]
        X_te = X[test_idx]
        nu_tr = cross_fit_nuisances(X_tr, T_tr, Y_tr, seed=fold_seed)

        tau_oof["S"][test_idx] = s_learner_cate(
            X_tr, T_tr, Y_tr, seed=fold_seed, X_eval=X_te,
        )
        tau_oof["T"][test_idx] = t_learner_cate(
            X_tr, T_tr, Y_tr, seed=fold_seed, X_eval=X_te,
        )
        tau_oof["DR"][test_idx] = dr_learner_cate(
            X_tr, T_tr, Y_tr, nuisances=nu_tr, seed=fold_seed, X_eval=X_te,
        )
        tau_oof["R"][test_idx] = r_learner_cate(
            X_tr, T_tr, Y_tr, nuisances=nu_tr, seed=fold_seed, X_eval=X_te,
        )
        cf_tau, _, _, _ = causal_forest_cate(
            X_tr, T_tr, Y_tr, nuisances=nu_tr, seed=fold_seed, X_eval=X_te,
        )
        tau_oof["CausalForest"][test_idx] = cf_tau

    return tau_oof


def best_linear_projection(
    tau_hat: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    nuisances: Nuisances | None = None,
    Y: np.ndarray | None = None,
    T: np.ndarray | None = None,
) -> pd.DataFrame:
    """OLS regression of τ̂(X) on selected X, with HC3 robust SEs.

    If (Y, T, nuisances) are provided, regress the DR pseudo-outcome instead of
    τ̂, which gives Semenova & Chernozhukov (2021) inference on the best linear
    projection of the true τ(X) onto the chosen features.
    """
    if nuisances is not None and Y is not None and T is not None:
        mu0, mu1, e = nuisances.mu0, nuisances.mu1, nuisances.e
        target = (
            mu1 - mu0
            + T * (Y - mu1) / e
            - (1 - T) * (Y - mu0) / (1 - e)
        )
    else:
        target = tau_hat

    X_design = sm.add_constant(X)
    model = sm.OLS(target, X_design).fit(cov_type="HC3")
    rows = []
    names = ["const"] + list(feature_names)
    for name, coef, se, p in zip(
        names, model.params, model.bse, model.pvalues
    ):
        rows.append({"feature": name, "coef": coef, "se": se, "pvalue": p})
    return pd.DataFrame(rows)


def gate_table(
    tau_hat: np.ndarray,
    groups: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    nuisances: Nuisances,
    group_labels: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Group ATE estimates from the AIPW influence function per group."""
    mu0, mu1, e = nuisances.mu0, nuisances.mu1, nuisances.e
    psi = (
        mu1 - mu0
        + T * (Y - mu1) / e
        - (1 - T) * (Y - mu0) / (1 - e)
    )
    unique = np.unique(groups)
    rows = []
    for g in unique:
        idx = groups == g
        n_g = int(idx.sum())
        est = float(psi[idx].mean())
        se = float(psi[idx].std(ddof=1) / np.sqrt(n_g))
        label = group_labels[int(g)] if group_labels is not None else f"g={g}"
        rows.append(
            {
                "group": label,
                "n": n_g,
                "gate": est,
                "se": se,
                "ci_low": est - 1.96 * se,
                "ci_high": est + 1.96 * se,
                "mean_tau_hat": float(tau_hat[idx].mean()),
            }
        )
    return pd.DataFrame(rows)


def rate_curve(
    tau_hat: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    nuisances: Nuisances,
    n_grid: int = 100,
    minimize: bool = True,
) -> pd.DataFrame:
    """Targeting Operator Characteristic + RATE (Yadlowsky et al., 2025).

    For each fraction q ∈ (0, 1], compute the average DR pseudo-outcome for the
    top-q fraction ranked by τ̂. When `minimize=True` (the default for our
    mortality outcome), higher impact = more negative τ̂, so the ranking is in
    ascending order of τ̂.
    """
    mu0, mu1, e = nuisances.mu0, nuisances.mu1, nuisances.e
    psi = (
        mu1 - mu0
        + T * (Y - mu1) / e
        - (1 - T) * (Y - mu0) / (1 - e)
    )
    overall = float(psi.mean())
    order = np.argsort(tau_hat) if minimize else np.argsort(-tau_hat)
    fractions = np.linspace(1.0 / n_grid, 1.0, n_grid)
    rows = []
    for q in fractions:
        k = max(1, int(round(q * len(psi))))
        top = order[:k]
        rows.append(
            {"fraction": q, "toc": float(psi[top].mean()), "overall_ate": overall}
        )
    return pd.DataFrame(rows)


def rate(toc_df: pd.DataFrame, minimize: bool = True) -> float:
    """RATE = integral over q of the targeting benefit.

    For a minimization outcome, useful prioritization makes TOC(q) MORE NEGATIVE
    than the overall ATE, so RATE = integral(overall - TOC) dq is the natural
    sign convention (positive means useful).
    """
    if minimize:
        diff = toc_df["overall_ate"].values - toc_df["toc"].values
    else:
        diff = toc_df["toc"].values - toc_df["overall_ate"].values
    q = toc_df["fraction"].values
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(diff, q))


# ---- CATE-adapted refutations -----------------------------------------------

@dataclass
class CATERefutationResult:
    test: str
    original_mean: float
    original_sd: float
    refuted_mean: float
    refuted_sd: float


def cate_placebo(
    fit_fn,
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_iter: int = 5,
    seed: int = 0,
) -> CATERefutationResult:
    """Permute T and refit the CATE estimator. The fitted τ̂(x) should collapse
    toward zero everywhere (not just on average).

    `fit_fn` is a callable (X, T, Y) -> τ̂(X).
    """
    tau_orig = fit_fn(X, T, Y)
    rng = np.random.default_rng(seed)
    refuted = []
    for i in range(n_iter):
        T_perm = rng.permutation(T)
        refuted.append(fit_fn(X, T_perm, Y))
    refuted = np.stack(refuted, axis=0)
    return CATERefutationResult(
        test="cate_placebo",
        original_mean=float(tau_orig.mean()),
        original_sd=float(tau_orig.std()),
        refuted_mean=float(refuted.mean()),
        refuted_sd=float(refuted.std()),
    )


def cate_random_common_cause(
    fit_fn,
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_iter: int = 3,
    seed: int = 0,
) -> CATERefutationResult:
    """Add an irrelevant N(0,1) covariate and refit. τ̂(X) should be ≈ unchanged."""
    tau_orig = fit_fn(X, T, Y)
    rng = np.random.default_rng(seed)
    refuted_means = []
    refuted_sds = []
    deltas = []
    for i in range(n_iter):
        Z = rng.normal(size=(X.shape[0], 1))
        tau_new = fit_fn(np.column_stack([X, Z]), T, Y)
        refuted_means.append(tau_new.mean())
        refuted_sds.append(tau_new.std())
        deltas.append(float(np.mean((tau_new - tau_orig) ** 2) ** 0.5))
    return CATERefutationResult(
        test=f"cate_random_common_cause (mean RMSE vs original = {np.mean(deltas):.4f})",
        original_mean=float(tau_orig.mean()),
        original_sd=float(tau_orig.std()),
        refuted_mean=float(np.mean(refuted_means)),
        refuted_sd=float(np.mean(refuted_sds)),
    )


def cate_subsample_stability(
    fit_fn,
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    fraction: float = 0.8,
    n_iter: int = 3,
    seed: int = 0,
) -> pd.DataFrame:
    """Refit on n_iter random subsamples. For each subsample, the fit_fn is called
    on the full X (after fitting on the subsample) so we get a pointwise SD over
    the full sample. fit_fn must accept (X_train, T_train, Y_train, X_eval) and
    return τ̂(X_eval); if only (X, T, Y) is supported, we call it on the
    subsample and pad with NaN.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = int(round(fraction * n))
    preds = np.full((n_iter, n), np.nan)
    for i in range(n_iter):
        idx = rng.choice(n, size=k, replace=False)
        try:
            preds[i] = fit_fn(X[idx], T[idx], Y[idx], X)
        except TypeError:
            preds[i, idx] = fit_fn(X[idx], T[idx], Y[idx])
    pointwise_sd = np.nanstd(preds, axis=0)
    return pd.DataFrame({
        "pair_index": np.arange(n),
        "pointwise_sd": pointwise_sd,
        "n_subsamples_covered": (~np.isnan(preds)).sum(axis=0),
    })
