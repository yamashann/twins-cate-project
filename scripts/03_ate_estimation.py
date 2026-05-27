"""ATE estimation: outcome regression and AIPW."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ate import aipw_ate, outcome_regression_ate
from src.data import load_twins
from src.nuisance import cross_fit_nuisances
from src.utils import ensure_dir


def main(seed: int = 0) -> None:
    res_dir = ensure_dir(ROOT / "results")
    data = load_twins(seed=seed)

    nu = cross_fit_nuisances(data.X, data.T, data.Y, seed=seed)
    or_res = outcome_regression_ate(data.X, data.T, data.Y, nuisances=nu, seed=seed)
    dr_res = aipw_ate(data.X, data.T, data.Y, nuisances=nu, seed=seed)

    truth = float(data.ite.mean())
    rows = []
    for r in (or_res, dr_res):
        rows.append({
            "method": r.method,
            "estimate": r.estimate,
            "se": r.se,
            "ci_low": r.ci_low,
            "ci_high": r.ci_high,
            "n": r.n,
        })
    rows.append({
        "method": "WithinPair (ground truth)",
        "estimate": truth,
        "se": float(data.ite.std(ddof=1) / (len(data.ite) ** 0.5)),
        "ci_low": None,
        "ci_high": None,
        "n": len(data.ite),
    })
    df = pd.DataFrame(rows)
    df.to_csv(res_dir / "03_ate.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
