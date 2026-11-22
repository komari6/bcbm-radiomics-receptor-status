"""
Step 17 — Generalization of the evaluation-optimism gap across public radiomics datasets
========================================================================================
Tests whether the single-split-vs-rigorous-CV "optimism gap" we observed in the
BCBM cohort is a *general* phenomenon, using the public radMLBench collection
(Demircioğlu) of radiomics datasets. CPU-only.

For each dataset, with a fixed leakage-safe light model (elastic-net in a
median-impute -> variance -> correlation(0.88) -> scale pipeline):
  - honest_cv      = mean AUROC over repeated stratified K-fold CV
  - single_split   = AUROC distribution over many random 80:20 splits
  - optimism_gap   = 95th percentile(single_split) - honest_cv   (a "lucky split")
  - leak_inflation = (feature selection on FULL data then CV) - honest_cv

Aggregated across datasets, this quantifies how much a single split and/or
preprocessing leakage overstate radiomics performance — the same mechanism that
made BCBM look promising (locked-test 0.71) under naive evaluation while nested
CV showed chance (~0.53).

Usage:
  pip install radMLBench
  python src/analysis/step_17_radmlbench_generalization.py [--max-datasets N] [--max-samples M]

Outputs:
  reports/figures/step13_publication_package/fig_optimism_generalization.{png,pdf}
  reports/tables/step17_optimism_gap.csv
  metadata/step17_optimism_summary.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

_REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = _REPO_ROOT / "bcbm_project"
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = Path("./bcbm_project").resolve()
STEP_NAME = "step_17_radmlbench_generalization"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "step13_publication_package"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
METADATA_DIR = PROJECT_ROOT / "metadata"
LOGS_DIR = PROJECT_ROOT / "logs"

GLOBAL_SEED = 42
N_SPLITS = 5
N_REPEATS = 5           # repeated CV for the honest estimate
N_SINGLE_SPLITS = 60    # random 80:20 splits for the single-split distribution
TOPK = 20               # features kept by SelectKBest in the (deliberately leaky) arm
MAX_FEATURES = 300      # label-free variance cap to bound saga runtime on wide datasets
MAX_ITER = 2000         # saga iterations (lowered for tractability across many datasets)


def setup_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(STEP_NAME); lg.setLevel(logging.INFO)
    if not lg.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        sh = logging.StreamHandler(); sh.setFormatter(fmt); lg.addHandler(sh)
        fh = logging.FileHandler(LOGS_DIR / f"{STEP_NAME}.log", encoding="utf-8"); fh.setFormatter(fmt); lg.addHandler(fh)
    return lg


def make_pipe(seed: int, k: Optional[int] = None) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median")),
             ("variance", VarianceThreshold(1e-8))]
    if k is not None:
        steps.append(("select", SelectKBest(f_classif, k="all" if k <= 0 else k)))
    steps += [("scaler", RobustScaler()),
              ("model", LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5,
                                           C=0.5, class_weight="balanced", max_iter=MAX_ITER, random_state=seed))]
    return Pipeline(steps)


def repeated_cv_auc(X, y, seed=GLOBAL_SEED) -> float:
    aucs = []
    for r in range(N_REPEATS):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed + r)
        oof = np.full(len(y), np.nan)
        for tr, va in skf.split(X, y):
            try:
                p = make_pipe(seed + r); p.fit(X[tr], y[tr]); oof[va] = p.predict_proba(X[va])[:, 1]
            except Exception:
                pass
        m = ~np.isnan(oof)
        if m.sum() and len(np.unique(y[m])) > 1:
            aucs.append(roc_auc_score(y[m], oof[m]))
    return float(np.mean(aucs)) if aucs else np.nan


def single_split_aucs(X, y) -> np.ndarray:
    out = []
    for i in range(N_SINGLE_SPLITS):
        try:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=GLOBAL_SEED + i)
            if len(np.unique(yte)) < 2:
                continue
            p = make_pipe(GLOBAL_SEED + i); p.fit(Xtr, ytr)
            out.append(roc_auc_score(yte, p.predict_proba(Xte)[:, 1]))
        except Exception:
            pass
    return np.array(out)


def leaky_cv_auc(X, y, seed=GLOBAL_SEED) -> float:
    """Deliberately leaky: SelectKBest on the FULL data, then CV (the classic mistake)."""
    try:
        sel = SelectKBest(f_classif, k=min(TOPK, X.shape[1])).fit(X, y)
        Xs = sel.transform(X)
    except Exception:
        return np.nan
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    for tr, va in skf.split(Xs, y):
        try:
            p = make_pipe(seed); p.fit(Xs[tr], y[tr]); oof[va] = p.predict_proba(Xs[va])[:, 1]
        except Exception:
            pass
    m = ~np.isnan(oof)
    return float(roc_auc_score(y[m], oof[m])) if m.sum() and len(np.unique(y[m])) > 1 else np.nan


def load_radmlbench(max_datasets: Optional[int], max_samples: int, logger):
    import radMLBench
    names = radMLBench.listDatasets()
    logger.info("radMLBench datasets available: %d", len(names))
    if max_datasets:
        names = names[:max_datasets]
    for name in names:
        try:
            d = radMLBench.loadData(name, return_X_y=False)
            df = d if isinstance(d, pd.DataFrame) else None
            if df is None:
                continue
            # radMLBench convention: 'Target' column is the binary label; drop ID columns
            if "Target" not in df.columns:
                continue
            y = pd.to_numeric(df["Target"], errors="coerce")
            Xdf = df.drop(columns=[c for c in df.columns if c.lower() in ("target", "id", "patient", "patientid")])
            Xdf = Xdf.apply(pd.to_numeric, errors="coerce")
            keep = y.notna()
            Xdf, y = Xdf[keep], y[keep].astype(int)
            if y.nunique() != 2 or len(y) < 40 or Xdf.shape[1] < 5:
                continue
            # Label-free cap on width to bound saga runtime: keep highest-variance columns.
            # Unsupervised (uses no labels) so it does not leak into the AUROC estimates.
            if Xdf.shape[1] > MAX_FEATURES:
                top = Xdf.var(numeric_only=True).sort_values(ascending=False).head(MAX_FEATURES).index
                Xdf = Xdf[top]
            if len(y) > max_samples:
                idx = (np.random.default_rng(GLOBAL_SEED).permutation(len(y)))[:max_samples]
                Xdf, y = Xdf.iloc[idx], y.iloc[idx]
            yield name, Xdf.to_numpy(dtype=float), y.to_numpy(dtype=int)
        except Exception as e:
            logger.warning("skip %s: %s", name, e)


def main():
    ap = argparse.ArgumentParser()
    # 18 = the compute-budget cut used for the published result (first 18 of the collection's
    # 50 datasets, in listing order). Pinned so the released code reproduces the paper's numbers;
    # pass None/0 to sweep the whole collection.
    ap.add_argument("--max-datasets", type=int, default=18)
    ap.add_argument("--max-samples", type=int, default=400)
    a = ap.parse_args()
    for d in (FIGURES_DIR, TABLES_DIR, METADATA_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    logger = setup_logger()
    logger.info("=" * 80); logger.info("Starting %s", STEP_NAME)
    try:
        import radMLBench  # noqa
    except Exception:
        logger.error("radMLBench not installed. Run: pip install radMLBench")
        raise SystemExit("radMLBench not installed")

    rows: List[Dict] = []
    for name, X, y in load_radmlbench(a.max_datasets, a.max_samples, logger):
        try:
            honest = repeated_cv_auc(X, y)
            ss = single_split_aucs(X, y)
            leaky = leaky_cv_auc(X, y)
        except Exception as e:                       # one bad dataset must not kill the whole run
            logger.warning("  compute failed for %s (%s); skipping", name[:22], e)
            continue
        if np.isnan(honest) or ss.size == 0:
            continue
        ss95 = float(np.percentile(ss, 95))
        rows.append({
            "dataset": name, "n": int(len(y)), "n_features": int(X.shape[1]),
            "prevalence": round(float(y.mean()), 3),
            "honest_cv_auc": round(honest, 4),
            "single_split_median": round(float(np.median(ss)), 4),
            "single_split_p95": round(ss95, 4),
            "single_split_sd": round(float(ss.std()), 4),
            "optimism_gap_p95_minus_honest": round(ss95 - honest, 4),
            "leaky_cv_auc": round(leaky, 4) if not np.isnan(leaky) else None,
            "leakage_inflation": round(leaky - honest, 4) if not np.isnan(leaky) else None,
        })
        logger.info("  %-22s n=%d honest=%.3f single_p95=%.3f gap=%.3f leak_infl=%s",
                    name[:22], len(y), honest, ss95, ss95 - honest,
                    f"{leaky-honest:.3f}" if not np.isnan(leaky) else "NA")

    if not rows:
        logger.error("No datasets processed."); raise SystemExit(1)
    res = pd.DataFrame(rows)
    res.to_csv(TABLES_DIR / "step17_optimism_gap.csv", index=False, encoding="utf-8-sig")

    gaps = res["optimism_gap_p95_minus_honest"].to_numpy()
    leaks = res["leakage_inflation"].dropna().to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    axes[0].hist(gaps, bins=15, color="#E0735B", alpha=0.8, edgecolor="white")
    axes[0].axvline(float(np.median(gaps)), color="black", ls="--", label=f"median {np.median(gaps):.2f}")
    axes[0].set_xlabel("Single-split optimism gap (95th-percentile − honest CV AUROC)")
    axes[0].set_ylabel("Datasets"); axes[0].set_title(f"A. Single-split optimism across {len(res)} radiomics datasets", fontweight="bold", fontsize=11)
    axes[0].legend()
    if leaks.size:
        axes[1].hist(leaks, bins=15, color="#2F3E55", alpha=0.8, edgecolor="white")
        axes[1].axvline(float(np.median(leaks)), color="black", ls="--", label=f"median {np.median(leaks):.2f}")
    axes[1].set_xlabel("Leakage inflation (feature selection on full data − honest CV AUROC)")
    axes[1].set_ylabel("Datasets"); axes[1].set_title("B. Preprocessing-leakage inflation", fontweight="bold", fontsize=11)
    axes[1].legend()
    fig.suptitle("Evaluation-optimism generalizes across public radiomics datasets (radMLBench)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURES_DIR / "fig_optimism_generalization.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig_optimism_generalization.pdf", bbox_inches="tight")
    plt.close(fig)

    summary = {"step": STEP_NAME, "n_datasets": int(len(res)),
               "median_single_split_optimism": round(float(np.median(gaps)), 4),
               "median_leakage_inflation": round(float(np.median(leaks)), 4) if leaks.size else None,
               # max_datasets/max_samples/max_features are logged too: they bound which
               # slice of the collection the reported numbers describe.
               "config": {"n_splits": N_SPLITS, "n_repeats": N_REPEATS, "n_single_splits": N_SINGLE_SPLITS,
                          "topk_leaky": TOPK, "max_datasets": a.max_datasets, "max_samples": a.max_samples,
                          "max_features": MAX_FEATURES},
               "datasets": rows}
    (METADATA_DIR / "step17_optimism_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Completed %s — %d datasets; median optimism gap %.3f.", STEP_NAME, len(res), float(np.median(gaps)))
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
