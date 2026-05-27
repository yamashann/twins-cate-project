"""Load and prepare the Louizos et al. Twins dataset.

The Twins dataset (Almond, Chay & Lee, 2005; preprocessed by Louizos et al., 2017)
contains 11,984 same-sex twin pairs in which both infants weigh under 2 kg. For each
pair we observe both potential outcomes (mortality of the heavier and the lighter
twin), which makes the dataset a standard benchmark for evaluating CATE estimators
against ground truth.

Following Louizos et al. (2017), an observational study is simulated by selecting
*one* twin per pair using a covariate-dependent treatment-assignment rule. The
unselected twin's outcome serves as the unobserved counterfactual and is used only
for validation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler

RAW_BASE_URL = (
    "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS"
)
RAW_FILES = {
    "X": "twin_pairs_X_3years_samesex.csv",
    "T": "twin_pairs_T_3years_samesex.csv",
    "Y": "twin_pairs_Y_3years_samesex.csv",
}
OPTIONAL_FILES = {
    "desc": "covar_desc.txt",
    "type": "covar_type.txt",
}

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


@dataclass
class TwinsData:
    """Bundle of arrays representing the simulated observational Twins study.

    X       : covariates, shape (n, d)
    T       : observed treatment (1 = heavier twin assigned), shape (n,)
    Y       : observed mortality, shape (n,)
    Y0, Y1  : counterfactual mortality (lighter, heavier), shape (n,)
    ite     : pair-level individual treatment effect Y1 - Y0, shape (n,)
    pair_id : original pair index, shape (n,)
    feature_names : column names of X
    """

    X: np.ndarray
    T: np.ndarray
    Y: np.ndarray
    Y0: np.ndarray
    Y1: np.ndarray
    ite: np.ndarray
    pair_id: np.ndarray
    feature_names: list[str]

    def as_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.X, columns=self.feature_names)
        df["T"] = self.T
        df["Y"] = self.Y
        df["Y0"] = self.Y0
        df["Y1"] = self.Y1
        df["ite"] = self.ite
        df["pair_id"] = self.pair_id
        return df


def download_twins(data_dir: Path = DEFAULT_DATA_DIR) -> None:
    """Download the raw Twins CSVs into `data_dir` if they are not already present."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for fname in RAW_FILES.values():
        target = data_dir / fname
        if target.exists():
            continue
        url = f"{RAW_BASE_URL}/{fname}"
        print(f"Downloading {url}")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        target.write_bytes(resp.content)
    for fname in OPTIONAL_FILES.values():
        target = data_dir / fname
        if target.exists():
            continue
        url = f"{RAW_BASE_URL}/{fname}"
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            target.write_bytes(resp.content)
        except Exception:
            pass  # metadata files are not required by the pipeline


def _load_raw(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X = pd.read_csv(data_dir / RAW_FILES["X"], index_col=0)
    T = pd.read_csv(data_dir / RAW_FILES["T"], index_col=0)  # dbirwt_0, dbirwt_1
    Y = pd.read_csv(data_dir / RAW_FILES["Y"], index_col=0)  # mort_0, mort_1
    return X, T, Y


NON_FEATURE_COLUMNS = {
    "Unnamed: 0",     # row-index artifact from the upstream CSV
    "infant_id_0",    # pair-level IDs, not predictive
    "infant_id_1",
}


def _drop_unusable_columns(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    # Drop ID / index columns that are not covariates.
    drop_cols = [c for c in X.columns if c in NON_FEATURE_COLUMNS]
    X = X.drop(columns=drop_cols)
    # Drop columns that are entirely NA.
    X = X.dropna(axis=1, how="all")
    # Median-impute the rest.
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    return X


def simulate_treatment_assignment(
    X: np.ndarray,
    Y0: np.ndarray,
    Y1: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """Simulate confounded treatment assignment, following Louizos et al. (2017).

    Draws a coefficient vector w ~ N(0, 0.1 * I_d), forms a logit
        p_i = sigmoid(z_i^T w + n_i)
    where z_i is the *full* standardized covariate vector and n_i ~ N(0, 0.1).
    The treatment is T_i ~ Bernoulli(p_i). This induces confounding because p
    depends on covariates that also drive Y; the unselected twin's outcome is
    reserved as ground truth in `TwinsData.Y0/Y1/ite`.

    Note: `Y0` and `Y1` are accepted in the signature to make the simulation
    contract explicit (the assignment knows what counterfactuals are being
    revealed) but are not used in the assignment itself, matching Louizos et
    al.'s setup where the assignment depends only on observed covariates.
    """
    del Y0, Y1  # not used; see docstring
    rng = np.random.default_rng(seed)
    z = StandardScaler().fit_transform(X)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    d = z.shape[1]
    w = rng.normal(loc=0.0, scale=0.1, size=d)
    noise = rng.normal(loc=0.0, scale=0.1, size=z.shape[0])
    logits = np.einsum("ij,j->i", z, w, optimize=True) + noise
    logits = np.clip(logits, -30.0, 30.0)
    p = 1.0 / (1.0 + np.exp(-logits))
    T = (rng.uniform(size=p.shape[0]) < p).astype(np.int64)
    return T


def load_twins(
    data_dir: Optional[Path] = None,
    seed: int = 0,
    download: bool = True,
) -> TwinsData:
    """Load the Twins dataset and produce the simulated observational sample.

    Parameters
    ----------
    data_dir : optional path
        Where the raw CSVs are stored. Defaults to <repo>/data/raw.
    seed : int
        RNG seed for the treatment-assignment simulation.
    download : bool
        If True, download the raw files when missing.
    """
    data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    if download:
        download_twins(data_dir)

    X_raw, T_raw, Y_raw = _load_raw(data_dir)

    # Keep pairs where both twins are alive at birth and weigh < 2 kg.
    # The "T" file stores birthweights (dbirwt_0 for the lighter, dbirwt_1 heavier).
    mask = (
        (T_raw["dbirwt_0"] < 2000)
        & (T_raw["dbirwt_1"] < 2000)
        & T_raw["dbirwt_0"].notna()
        & T_raw["dbirwt_1"].notna()
        & Y_raw["mort_0"].notna()
        & Y_raw["mort_1"].notna()
    )
    X_raw = X_raw.loc[mask]
    T_raw = T_raw.loc[mask]
    Y_raw = Y_raw.loc[mask]

    # Align indices, drop unusable columns, impute.
    X_clean = _drop_unusable_columns(X_raw)
    feature_names = list(X_clean.columns)
    Xa = X_clean.to_numpy(dtype=np.float64)

    Y0 = Y_raw["mort_0"].to_numpy(dtype=np.float64)
    Y1 = Y_raw["mort_1"].to_numpy(dtype=np.float64)
    ite = Y1 - Y0
    pair_id = X_raw.index.to_numpy()

    T = simulate_treatment_assignment(Xa, Y0, Y1, seed=seed)
    Y = np.where(T == 1, Y1, Y0)

    return TwinsData(
        X=Xa,
        T=T,
        Y=Y,
        Y0=Y0,
        Y1=Y1,
        ite=ite,
        pair_id=pair_id,
        feature_names=feature_names,
    )


if __name__ == "__main__":
    d = load_twins()
    print(f"n = {len(d.Y)}, d = {d.X.shape[1]}")
    print(f"P(T=1) = {d.T.mean():.3f}")
    print(f"P(Y=1) = {d.Y.mean():.4f}")
    print(f"True ATE (Y1 - Y0) = {d.ite.mean():.4f}")
