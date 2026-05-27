"""Cross-fit nuisance estimators reused by ATE and CATE methods.

Both AIPW (for the ATE) and the orthogonalized CATE learners (DR-learner, R-learner)
need:
    mu_t(x) = E[Y | X=x, T=t]   for t = 0, 1
    e(x)    = P(T=1 | X=x)

These are estimated with K-fold cross-fitting so that, for every observation i, the
nuisance prediction at i is produced by a model fit on a fold that excludes i. This
is what gives the downstream estimators their consistency / asymptotic-normality
guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class Nuisances:
    mu0: np.ndarray
    mu1: np.ndarray
    e: np.ndarray


def cross_fit_nuisances(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_splits: int = 5,
    seed: int = 0,
    propensity: str = "logreg",
    outcome: str = "gbm",
    clip: float = 1e-2,
) -> Nuisances:
    n = len(Y)
    mu0 = np.zeros(n)
    mu1 = np.zeros(n)
    e = np.zeros(n)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(X, T):
        X_tr, T_tr, Y_tr = X[train_idx], T[train_idx], Y[train_idx]
        X_te = X[test_idx]

        if propensity == "logreg":
            ps = Pipeline([
                ("scale", StandardScaler()),
                ("lr", LogisticRegression(max_iter=5000, C=1.0, solver="lbfgs")),
            ])
        else:
            ps = GradientBoostingClassifier(random_state=seed)
        ps.fit(X_tr, T_tr)
        e[test_idx] = ps.predict_proba(X_te)[:, 1]

        idx0 = T_tr == 0
        idx1 = T_tr == 1
        if outcome == "gbm":
            m0 = GradientBoostingRegressor(random_state=seed)
            m1 = GradientBoostingRegressor(random_state=seed)
        else:
            from sklearn.linear_model import LogisticRegression as LR

            m0 = LR(max_iter=2000)
            m1 = LR(max_iter=2000)
        m0.fit(X_tr[idx0], Y_tr[idx0])
        m1.fit(X_tr[idx1], Y_tr[idx1])
        mu0[test_idx] = _predict(m0, X_te)
        mu1[test_idx] = _predict(m1, X_te)

    e = np.clip(e, clip, 1.0 - clip)
    mu0 = np.clip(mu0, 0.0, 1.0)
    mu1 = np.clip(mu1, 0.0, 1.0)
    return Nuisances(mu0=mu0, mu1=mu1, e=e)


def _predict(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)
