"""Exploratory data analysis and overlap / balance diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from .data import TwinsData
from .utils import ensure_dir


def summary_table(data: TwinsData) -> pd.DataFrame:
    rows = [
        ("n (observations)", len(data.Y)),
        ("n pairs", data.X.shape[0]),
        ("d (covariates)", data.X.shape[1]),
        ("P(T=1)", float(data.T.mean())),
        ("P(Y=1)", float(data.Y.mean())),
        ("P(Y=1 | T=1)", float(data.Y[data.T == 1].mean())),
        ("P(Y=1 | T=0)", float(data.Y[data.T == 0].mean())),
        ("Naive diff (treated - control)", float(
            data.Y[data.T == 1].mean() - data.Y[data.T == 0].mean()
        )),
        ("Within-pair ATE (ground truth)", float(data.ite.mean())),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def standardized_mean_differences(data: TwinsData) -> pd.DataFrame:
    treated = data.X[data.T == 1]
    control = data.X[data.T == 0]
    mu_t = treated.mean(axis=0)
    mu_c = control.mean(axis=0)
    var_t = treated.var(axis=0, ddof=1)
    var_c = control.var(axis=0, ddof=1)
    pooled_sd = np.sqrt((var_t + var_c) / 2.0)
    pooled_sd = np.where(pooled_sd == 0, np.nan, pooled_sd)
    smd = (mu_t - mu_c) / pooled_sd
    out = pd.DataFrame({
        "feature": data.feature_names,
        "mean_treated": mu_t,
        "mean_control": mu_c,
        "smd": smd,
        "abs_smd": np.abs(smd),
    })
    return out.sort_values("abs_smd", ascending=False)


def estimate_propensity(data: TwinsData, seed: int = 0) -> np.ndarray:
    """Cross-fit logistic-regression propensity scores."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    lr = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=5000, random_state=seed,
        )),
    ])
    e_hat = cross_val_predict(
        lr, data.X, data.T, cv=5, method="predict_proba"
    )[:, 1]
    return np.clip(e_hat, 1e-3, 1.0 - 1e-3)


def overlap_plot(
    e_hat: np.ndarray,
    T: np.ndarray,
    out_path: Path,
    bins: int = 40,
) -> None:
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(e_hat[T == 1], bins=bins, alpha=0.6, label="treated (heavier)", density=True)
    ax.hist(e_hat[T == 0], bins=bins, alpha=0.6, label="control (lighter)", density=True)
    ax.set_xlabel("estimated propensity $\\hat{e}(x)$")
    ax.set_ylabel("density")
    ax.set_title("Propensity-score overlap")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def smd_plot(smd_table: pd.DataFrame, out_path: Path, top: int = 20) -> None:
    ensure_dir(out_path.parent)
    df = smd_table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, max(4, 0.25 * len(df))))
    ax.barh(df["feature"], df["smd"])
    ax.axvline(0.1, color="grey", linestyle=":", linewidth=1)
    ax.axvline(-0.1, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("standardized mean difference (treated - control)")
    ax.set_title(f"Top {top} covariate imbalances")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def mortality_by_treatment_plot(data: TwinsData, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    counts = pd.DataFrame({
        "group": ["lighter (T=0)", "heavier (T=1)"],
        "mortality": [data.Y[data.T == 0].mean(), data.Y[data.T == 1].mean()],
    })
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.bar(counts["group"], counts["mortality"], color=["#a6a6d8", "#5a5aa8"])
    ax.set_ylabel("one-year mortality rate")
    ax.set_title("Observed mortality by treatment group")
    for i, v in enumerate(counts["mortality"]):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
