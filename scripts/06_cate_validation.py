"""CATE validation: ground-truth comparison, R-loss / DR-MSE, calibration, RATE,
GATEs, best linear projection, placebo refutation."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cate import (
    causal_forest_cate,
    dr_learner_cate,
    fit_all_cates,
    r_learner_cate,
    s_learner_cate,
    t_learner_cate,
)
from src.cate_validation import (
    best_linear_projection,
    calibration_table,
    cate_placebo,
    cate_random_common_cause,
    cate_subsample_stability,
    dr_score_mse,
    gate_table,
    ground_truth_scores,
    heldout_cate_predictions,
    pehe,
    r_loss,
    rate,
    rate_curve,
)
from src.data import load_twins
from src.nuisance import cross_fit_nuisances
from src.utils import ensure_dir


def main(seed: int = 0) -> None:
    fig_dir = ensure_dir(ROOT / "figures")
    res_dir = ensure_dir(ROOT / "results")

    data = load_twins(seed=seed)
    nu = cross_fit_nuisances(data.X, data.T, data.Y, seed=seed)
    estimates = fit_all_cates(data.X, data.T, data.Y, nuisances=nu, seed=seed)

    methods = {
        "S": estimates.s_learner,
        "T": estimates.t_learner,
        "DR": estimates.dr_learner,
        "R": estimates.r_learner,
        "CausalForest": estimates.causal_forest,
    }

    # 1. Ground-truth scores
    gt = ground_truth_scores(methods, data.ite)
    gt.to_csv(res_dir / "06_ground_truth_scores.csv", index=False)
    print("Ground-truth (PEHE / rank corr):")
    print(gt.to_string(index=False))

    # 2. R-loss and DR-score MSE, evaluated on CATE-stage out-of-fold predictions.
    heldout_methods = heldout_cate_predictions(
        data.X, data.T, data.Y, n_splits=5, seed=seed,
    )
    rows = []
    for name, tau in heldout_methods.items():
        rows.append({
            "method": name,
            "evaluation": "5-fold CATE-stage heldout",
            "r_loss": r_loss(tau, data.Y, data.T, nu),
            "dr_score_mse": dr_score_mse(tau, data.Y, data.T, nu),
        })
    pd.DataFrame(rows).to_csv(res_dir / "06_model_selection.csv", index=False)

    # 3. Calibration for the best ground-truth method
    best_name = gt.sort_values("pehe").iloc[0]["method"]
    best_tau = methods[best_name]
    cal = calibration_table(best_tau, data.ite, n_bins=10)
    cal.to_csv(res_dir / "06_calibration.csv", index=False)
    _calibration_plot(cal, fig_dir / "06_calibration.png", best_name)

    # 4. RATE curve / Qini-style
    toc = rate_curve(best_tau, data.Y, data.T, nu)
    toc.to_csv(res_dir / "06_rate_curve.csv", index=False)
    _rate_plot(toc, fig_dir / "06_rate_curve.png", best_name)
    rate_value = rate(toc)
    print(f"\nRATE (top-q vs overall, {best_name}) = {rate_value:.4f}")

    # 5. GATEs by gestational-age quartile (collapses to tertiles for this dataset)
    # Report BOTH the AIPW GATE (the observational-data estimate) and the
    # within-pair true GATE (design-based ground truth), so the report can
    # compare and call out any divergence.
    ga_col = _find_col(data.feature_names, ["gestat", "gest_age", "ga"])
    if ga_col is not None:
        ga = data.X[:, ga_col]
        quartiles = np.asarray(pd.qcut(ga, q=4, labels=False, duplicates="drop"))
        labels = [f"Q{i+1} (gest. age)" for i in range(int(np.nanmax(quartiles)) + 1)]
        gate = gate_table(best_tau, quartiles, data.Y, data.T, nu, labels)

        # Within-pair true GATE per tertile, with SE = sd(ITE)/sqrt(n).
        truth_rows = []
        for q_idx, label in enumerate(labels):
            mask = quartiles == q_idx
            n_q = int(mask.sum())
            mean_ite = float(data.ite[mask].mean())
            sd_ite = float(data.ite[mask].std(ddof=1))
            se = sd_ite / np.sqrt(n_q)
            truth_rows.append({
                "group": label,
                "true_gate": mean_ite,
                "true_se": se,
                "true_ci_low": mean_ite - 1.96 * se,
                "true_ci_high": mean_ite + 1.96 * se,
            })
        truth_df = pd.DataFrame(truth_rows)
        gate = gate.merge(truth_df, on="group")
        gate.to_csv(res_dir / "06_gate_by_gestational_age.csv", index=False)
        print("\nGATEs by gestational-age tertile (AIPW vs. within-pair truth):")
        print(gate.to_string(index=False))

    # 5b. GATEs by prenatal-care intensity quartile (proposal asked for this too).
    pc_col = _find_col(data.feature_names, ["nprevistq", "nprevist"])
    if pc_col is not None:
        pc = data.X[:, pc_col]
        pc_bins = np.asarray(pd.qcut(pc, q=4, labels=False, duplicates="drop"))
        pc_labels = [
            f"Q{i+1} (prenatal visits)"
            for i in range(int(np.nanmax(pc_bins)) + 1)
        ]
        gate_pc = gate_table(best_tau, pc_bins, data.Y, data.T, nu, pc_labels)
        truth_rows = []
        for q_idx, label in enumerate(pc_labels):
            mask = pc_bins == q_idx
            n_q = int(mask.sum())
            mean_ite = float(data.ite[mask].mean())
            sd_ite = float(data.ite[mask].std(ddof=1))
            se = sd_ite / np.sqrt(n_q)
            truth_rows.append({
                "group": label,
                "true_gate": mean_ite, "true_se": se,
                "true_ci_low": mean_ite - 1.96 * se,
                "true_ci_high": mean_ite + 1.96 * se,
            })
        gate_pc = gate_pc.merge(pd.DataFrame(truth_rows), on="group")
        gate_pc.to_csv(res_dir / "06_gate_by_prenatal_care.csv", index=False)
        print("\nGATEs by prenatal-care intensity quartile (AIPW vs. within-pair truth):")
        print(gate_pc.to_string(index=False))

    # 6. Best linear projection on a clinically meaningful subset of modifiers.
    # Report two versions side-by-side: BLP on the AIPW pseudo-outcome (the
    # standard observational-data estimator) and BLP on the within-pair TRUE
    # ITE (ground truth, only available on the Twins benchmark). This is the
    # subgroup analog of the GATE comparison in step 5.
    import statsmodels.api as sm

    blp_targets = [
        "gestat10",      # gestational age (10-week bin)
        "nprevistq",     # number of prenatal visits (quartile)
        "mager8",        # maternal age (8-level)
        "meduc6",        # maternal education (6-level)
        "anemia", "diabetes", "chyper", "preterm",
    ]
    blp_idx = [data.feature_names.index(c) for c in blp_targets if c in data.feature_names]
    blp_names = [data.feature_names[i] for i in blp_idx]

    blp_aipw = best_linear_projection(
        best_tau, data.X[:, blp_idx], blp_names,
        nuisances=nu, Y=data.Y, T=data.T,
    ).rename(columns={"coef": "aipw_coef", "se": "aipw_se", "pvalue": "aipw_p"})

    # Truth BLP: regress within-pair ITE on the same features, HC3 SEs.
    X_design = sm.add_constant(data.X[:, blp_idx])
    truth_fit = sm.OLS(data.ite, X_design).fit(cov_type="HC3")
    truth_blp = pd.DataFrame({
        "feature": ["const"] + blp_names,
        "truth_coef": truth_fit.params,
        "truth_se": truth_fit.bse,
        "truth_p": truth_fit.pvalues,
    })
    blp = blp_aipw.merge(truth_blp, on="feature")
    blp.to_csv(res_dir / "06_best_linear_projection.csv", index=False)
    print("\nBest linear projection (AIPW pseudo-outcome vs. within-pair truth):")
    print(blp.to_string(index=False))

    # 6b. PEHE benchmark: how does each CATE method compare to "predict the mean"?
    const_pehe = float(np.sqrt(((data.ite - data.ite.mean()) ** 2).mean()))
    print(f"\nPEHE baseline (constant predictor = mean ITE) = {const_pehe:.4f}")
    print(f"  → reductions vs. baseline:")
    for name, tau in methods.items():
        method_pehe = float(np.sqrt(((tau - data.ite) ** 2).mean()))
        print(f"    {name:12s}: PEHE = {method_pehe:.4f}, "
              f"{(const_pehe - method_pehe) / const_pehe * 100:+.1f}% vs baseline")

    # 6c. Variance decomposition: lower bound on true sd(CATE) by binning ITE by tau_hat.
    print("\nLower bound on true sd(CATE) (between-bin variance, ranked by tau_CF, 50 bins):")
    order = np.argsort(estimates.causal_forest)
    bins = np.array_split(order, 50)
    bin_means = np.array([data.ite[idx].mean() for idx in bins])
    bin_weights = np.array([len(idx) for idx in bins])
    between_var = float(np.average((bin_means - data.ite.mean()) ** 2, weights=bin_weights))
    print(f"  Var[ITE] = {data.ite.var():.4f}")
    print(f"  Between-bin Var of TRUE ITE = {between_var:.4f}  → sd ≥ {np.sqrt(between_var):.4f}")
    print(f"  CausalForest predicted sd = 0.022; DR predicted sd = 0.168")

    # 6d. Causal forest feature importance (proposal: "variable importance for
    # heterogeneity"). econml's CausalForestDML exposes feature_importances_.
    cf_model = estimates.causal_forest_model
    if cf_model is not None and hasattr(cf_model, "feature_importances_"):
        importances = np.asarray(cf_model.feature_importances_)
        fi = pd.DataFrame({
            "feature": data.feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)
        fi.to_csv(res_dir / "06_causal_forest_feature_importance.csv", index=False)
        print("\nCausal forest feature importance (top 15):")
        print(fi.head(15).to_string(index=False))
        _feature_importance_plot(fi.head(15), fig_dir / "06_feature_importance.png")

    # 6e. Causal forest pointwise CIs for the headline gestational-age subgroups.
    # Proposal: "pointwise confidence intervals for τ(x)". Report mean CI width
    # by tertile and the fraction of units whose CI excludes 0.
    if ga_col is not None:
        rows = []
        for q_idx, label in enumerate(labels):
            mask = quartiles == q_idx
            cf_tau = estimates.causal_forest[mask]
            cf_lo = estimates.causal_forest_lo[mask]
            cf_hi = estimates.causal_forest_hi[mask]
            excludes_zero = float(np.mean((cf_lo > 0) | (cf_hi < 0)))
            rows.append({
                "group": label, "n": int(mask.sum()),
                "mean_tau_hat": float(cf_tau.mean()),
                "mean_ci_width": float((cf_hi - cf_lo).mean()),
                "frac_ci_excludes_zero": excludes_zero,
            })
        cf_ci = pd.DataFrame(rows)
        cf_ci.to_csv(res_dir / "06_causal_forest_pointwise_cis.csv", index=False)
        print("\nCausal forest pointwise CIs by gestational-age tertile:")
        print(cf_ci.to_string(index=False))

    # 7. CATE-adapted refutations (proposal Section 5): placebo, random common
    # cause, and subsample stability. We use the DR-learner for these because
    # it is the fastest of the orthogonalized estimators.
    def fit_dr(X, T, Y, X_eval=None):
        nu = cross_fit_nuisances(X, T, Y, seed=seed)
        return dr_learner_cate(X, T, Y, nuisances=nu, seed=seed, X_eval=X_eval)

    placebo = cate_placebo(fit_dr, data.X, data.T, data.Y, n_iter=3, seed=seed)
    pd.DataFrame([placebo.__dict__]).to_csv(res_dir / "06_cate_placebo.csv", index=False)
    print("\nCATE placebo (permuted T):")
    print(placebo)

    rcc = cate_random_common_cause(fit_dr, data.X, data.T, data.Y, n_iter=3, seed=seed)
    pd.DataFrame([rcc.__dict__]).to_csv(res_dir / "06_cate_random_common_cause.csv", index=False)
    print("\nCATE random common cause (added irrelevant N(0,1) covariate):")
    print(rcc)

    stab = cate_subsample_stability(fit_dr, data.X, data.T, data.Y, fraction=0.8, n_iter=3, seed=seed)
    stab.to_csv(res_dir / "06_cate_subsample_stability.csv", index=False)
    median_sd = float(stab["pointwise_sd"].median())
    print(f"\nCATE subsample stability: median pointwise SD across 3 80%-resamples = {median_sd:.4f}")


def _find_col(names, needles):
    lname = [n.lower() for n in names]
    for needle in needles:
        for i, n in enumerate(lname):
            if needle in n:
                return i
    return None


def _calibration_plot(df, out_path, method_name):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(df["mean_predicted"], df["mean_true_ite"], "o-")
    lim = [
        float(min(df["mean_predicted"].min(), df["mean_true_ite"].min())),
        float(max(df["mean_predicted"].max(), df["mean_true_ite"].max())),
    ]
    ax.plot(lim, lim, "--", color="grey", label="y = x")
    ax.set_xlabel("decile mean of $\\hat{\\tau}(x)$")
    ax.set_ylabel("decile mean of true ITE")
    ax.set_title(f"Calibration vs. twin-pair benchmark ({method_name})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _rate_plot(df, out_path, method_name):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(df["fraction"], df["toc"], label="TOC: mean DR-score in top q")
    ax.axhline(df["overall_ate"].iloc[0], color="grey", linestyle="--", label="overall ATE")
    ax.set_xlabel("fraction prioritized by $\\hat{\\tau}(x)$")
    ax.set_ylabel("average treatment effect in prioritized group")
    ax.set_title(f"Targeting operator characteristic ({method_name})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _feature_importance_plot(df, out_path):
    df = df.iloc[::-1]  # ascending for horizontal bar
    fig, ax = plt.subplots(figsize=(6, max(4, 0.3 * len(df))))
    ax.barh(df["feature"], df["importance"])
    ax.set_xlabel("causal-forest feature importance")
    ax.set_title("Causal forest: top features driving heterogeneity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
