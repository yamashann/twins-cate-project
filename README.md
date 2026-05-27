# Twins CATE Project

Code for my final project in Causal Models in Data Science. The project
estimates the effect of being the heavier twin on first-year infant mortality,
using the Twins dataset (Louizos et al., 2017).

Included:

- **ATE estimation** by outcome regression (g-formula plug-in) and AIPW (doubly
  robust).
- **CATE estimation** by S-, T-, DR-, R-learner, and a causal forest (econml).
- **ATE refutation** (placebo treatment, random common cause, subset re-estimation).
- **CATE validation** against the within-pair design-based ground truth, plus
  5-fold held-out model selection (R-loss, DR-score MSE), calibration plots,
  GATEs by gestational-age quartile, the Semenova–Chernozhukov best linear
  projection, a RATE / TOC curve, and CATE-level refutations.
- **Sensitivity analysis** to a single simulated unmeasured confounder.

The full report is at [`report/final_report.md`](report/final_report.md).

## Repository layout

```
twins-cate-project/
├── README.md                  # this file
├── requirements.txt
├── data/
│   └── raw/                   # downloaded CSVs (auto, gitignored)
├── src/
│   ├── data.py                # download, filter, simulate observational sample
│   ├── eda.py                 # summary table, SMD, overlap, mortality plots
│   ├── dag.py                 # build + render the DAG
│   ├── nuisance.py            # 5-fold cross-fit μ̂_t, ê
│   ├── ate.py                 # outcome regression, AIPW
│   ├── cate.py                # S/T/DR/R-learner + causal forest
│   ├── refutation.py          # ATE refutation tests
│   ├── cate_validation.py     # PEHE, R-loss, DR-MSE, GATE, BLP, RATE, placebo
│   ├── sensitivity.py         # unmeasured-confounder grid + contour
│   └── utils.py
├── scripts/
│   ├── 01_eda.py              # EDA + overlap diagnostics
│   ├── 02_dag.py              # render the DAG
│   ├── 03_ate_estimation.py   # ATE point estimates and CIs
│   ├── 04_cate_estimation.py  # fit all CATE learners
│   ├── 05_ate_refutation.py   # placebo / random covariate / subset
│   ├── 06_cate_validation.py  # ground truth + R-loss + GATE + BLP + RATE
│   ├── 07_sensitivity.py      # unmeasured-confounder sweep
│   └── run_all.py             # runs 01–07 in order
├── figures/                   # generated PNGs (gitignored)
├── results/                   # generated CSVs (gitignored)
└── report/
    └── final_report.md
```

## Setup

```bash
# 1. Clone
git clone https://github.com/yamashann/twins-cate-project.git
cd twins-cate-project

# 2. Create and activate a virtual environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

`econml` requires a C/C++ toolchain; if installation fails on macOS, run
`xcode-select --install` first. On Linux, `apt install build-essential` is
typically enough.

## Reproducing the results

The full pipeline:

```bash
python scripts/run_all.py
```

This:

1. Downloads the raw Twins CSVs from the AMLab-Amsterdam/CEVAE repository into
   `data/raw/` (only on the first run; cached afterward).
2. Filters to same-sex pairs with both birthweights < 2000 g and both outcomes
   present; median-imputes missing covariates.
3. Simulates a covariate-dependent treatment assignment to induce confounding.
4. Runs scripts 01–07 sequentially.

Each script can also be run individually:

```bash
python scripts/01_eda.py
python scripts/02_dag.py
python scripts/03_ate_estimation.py
python scripts/04_cate_estimation.py
python scripts/05_ate_refutation.py
python scripts/06_cate_validation.py
python scripts/07_sensitivity.py
```

Outputs are written to:

- `figures/` — PNGs (overlap, SMD, DAG, calibration, RATE curve, sensitivity
  contour).
- `results/` — CSVs (summary table, SMD, ATE estimates with CIs, CATE
  predictions, ground-truth scores, model-selection table, GATEs, BLP,
  refutations, sensitivity grid).

The exact mapping between scripts and the tables/figures in the report is:

| Report table/figure | Script | Output file |
|---|---|---|
| Section 2.1 summary | `01_eda.py` | `results/01_summary.csv` |
| Figure 1 (overlap) | `01_eda.py` | `figures/01_overlap.png` |
| Figure 2 (SMD) | `01_eda.py` | `figures/01_smd_top20.png` |
| Figure 3 (DAG) | `02_dag.py` | `figures/02_dag.png` |
| Table 1 (ATE) | `03_ate_estimation.py` | `results/03_ate.csv` |
| Table 2 (CATE summary) | `04_cate_estimation.py` | `results/04_cate_summary.csv` |
| Section 5.1 refutations | `05_ate_refutation.py` | `results/05_refutations.csv` |
| Table 3 (PEHE etc.) | `06_cate_validation.py` | `results/06_ground_truth_scores.csv` |
| Table 4 (held-out R-loss / DR-MSE) | `06_cate_validation.py` | `results/06_model_selection.csv` |
| Figure 4 (calibration) | `06_cate_validation.py` | `figures/06_calibration.png` |
| Table 5 (gest-age GATE) | `06_cate_validation.py` | `results/06_gate_by_gestational_age.csv` |
| §5.6 (prenatal-care GATE) | `06_cate_validation.py` | `results/06_gate_by_prenatal_care.csv` |
| Table 6 (BLP) | `06_cate_validation.py` | `results/06_best_linear_projection.csv` |
| Figure 5 (CF feature importance) | `06_cate_validation.py` | `results/06_causal_forest_feature_importance.csv` + `figures/06_feature_importance.png` |
| §5.8 (CF pointwise CIs) | `06_cate_validation.py` | `results/06_causal_forest_pointwise_cis.csv` |
| Figure 6 / §5.9 (RATE) | `06_cate_validation.py` | `figures/06_rate_curve.png` |
| §5.10 CATE refutations | `06_cate_validation.py` | `results/06_cate_placebo.csv`, `06_cate_random_common_cause.csv`, `06_cate_subsample_stability.csv` |
| Figure 7 / §5.11 sensitivity | `07_sensitivity.py` | `results/07_sensitivity.csv` + `figures/07_sensitivity.png` |

All scripts accept a `seed` keyword (default 0) and seed both the
treatment-assignment simulation and all sklearn / econml estimators. Running

```bash
python scripts/run_all.py
```

twice with the same seed produces identical CSVs and figures.

## Data source and license

Raw data come from the publicly available
[AMLab-Amsterdam/CEVAE](https://github.com/AMLab-Amsterdam/CEVAE) repository,
preprocessed by Louizos et al. (2017) from the Almond–Chay–Lee (2005) Twins
mortality data. The downloader (`src/data.py`) fetches:

- `twin_pairs_X_3years_samesex.csv` — covariates
- `twin_pairs_T_3years_samesex.csv` — birth weights (used to construct T)
- `twin_pairs_Y_3years_samesex.csv` — mortality outcomes
- `covar_desc.txt`, `covar_type.txt` — covariate metadata

See those repositories for the original data license.
