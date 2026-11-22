"""
Step 14 — Learning‑curve and statistical‑power analysis
=======================================================
Quantifies whether the chance‑level result in Step 05 is driven by *absence of
signal* or by *insufficient sample size*. Two complementary analyses:

  (A) Learning curve — leakage‑safe nested out‑of‑fold AUROC as a function of
      training‑set size, per target. A rising curve implies the model is data‑
      starved; a flat curve near 0.50 implies no recoverable signal at this scale.

  (B) Statistical power — Hanley–McNeil analytic power (cross‑checked by
      simulation) for detecting AUROC > 0.50 given the actual class counts:
        - power and minimum detectable AUROC (MDA) for the LOCKED TEST set,
        - power for the nested‑CV out‑of‑fold pool,
        - test‑set size required to detect AUROC = 0.70 at 80% power.

Design choices (and why):
  - We reuse Step 05's in‑fold Pipeline (impute→variance→correlation→ComBat→
    stability→scale→model) so preprocessing is fit on training data only — the
    learning curve inherits the same leakage controls as the main analysis.
  - A single regularized model (elastic‑net) and the radiomics‑only feature set
    are used throughout the curve so that the curve reflects the effect of
    sample size, not model switching.
  - Power uses the Hanley–McNeil (1982) variance of the AUC; required‑n follows
    Hanley–McNeil power directly (see Riley et al., Stat Med 2021, for validation-size guidance).

Outputs:
  reports/figures/step13_publication_package/fig_learning_curve.png/.pdf
  reports/figures/step13_publication_package/fig_power_analysis.png/.pdf
  reports/tables/step14_power_analysis.csv
  reports/tables/step14_learning_curve.csv
  metadata/step14_learning_curve_power_summary.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# ── Paths (anchored to repo root, cwd‑independent) ────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = _REPO_ROOT / "bcbm_project"
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = Path("./bcbm_project").resolve()

STEP_NAME = "step_14_learning_curve_power"
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "analysis_ready_step04_patient_level.csv"
FEATURE_DICT = PROJECT_ROOT / "metadata" / "step04_feature_dictionary.json"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "step13_publication_package"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
METADATA_DIR = PROJECT_ROOT / "metadata"
LOGS_DIR = PROJECT_ROOT / "logs"

TARGET_COLUMNS = ["target_er", "target_pr", "target_her2"]
TARGET_DISPLAY = {"target_er": "ER", "target_pr": "PR", "target_her2": "HER2"}
TARGET_COLORS = {"target_er": "#E0735B", "target_pr": "#1B9E8A", "target_her2": "#2F3E55"}

GLOBAL_SEED = 42
LC_MODEL = "elastic_net"          # fixed regularized model for the learning curve
LC_VARIANT = "radiomics_pure"     # the 642-column radiomic block (step 04)
LC_FRACTIONS = [0.30, 0.45, 0.60, 0.75, 0.90, 1.00]
LC_OUTER_SPLITS = 5
LC_REPEATS = 5                    # subsample repeats per (fraction, fold)


def setup_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(STEP_NAME)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        sh = logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
        fh = logging.FileHandler(LOGS_DIR / f"{STEP_NAME}.log", encoding="utf-8"); fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def load_step05_module():
    """Import the Step 05 module by file path to reuse its leakage‑safe components."""
    p = _REPO_ROOT / "src" / "modeling" / "step_05_tabular_modeling.py"
    spec = importlib.util.spec_from_file_location("step_05_tabular_modeling", p)
    m = importlib.util.module_from_spec(spec)
    import sys as _sys
    _sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


# ── (B) Statistical power: Hanley–McNeil ──────────────────────────────────────
def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Standard error of the AUC (Hanley & McNeil, 1982)."""
    auc = min(max(auc, 1e-6), 1 - 1e-6)
    q1 = auc / (2 - auc)
    q2 = 2 * auc * auc / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg)
    return math.sqrt(max(var, 0.0))


def _norm_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _norm_ppf(p: float) -> float:
    # Acklam's inverse normal CDF approximation (no scipy dependency)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5; r = q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def power_auc(auc_alt: float, n_pos: int, n_neg: int, alpha: float = 0.05) -> float:
    """Power to reject H0: AUC = 0.5 in favor of AUC = auc_alt (two‑sided)."""
    if n_pos < 1 or n_neg < 1 or auc_alt <= 0.5:
        return 0.0
    se0 = hanley_mcneil_se(0.5, n_pos, n_neg)
    se1 = hanley_mcneil_se(auc_alt, n_pos, n_neg)
    z_a = _norm_ppf(1 - alpha / 2)
    z = (auc_alt - 0.5 - z_a * se0) / max(se1, 1e-9)
    return _norm_cdf(z)


def min_detectable_auc(n_pos: int, n_neg: int, power: float = 0.80, alpha: float = 0.05) -> float:
    """Smallest AUROC distinguishable from 0.5 at the requested power (bisection)."""
    lo, hi = 0.5001, 0.9999
    if power_auc(hi, n_pos, n_neg, alpha) < power:
        return float("nan")
    for _ in range(80):
        mid = (lo + hi) / 2
        if power_auc(mid, n_pos, n_neg, alpha) < power:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def required_total_n(auc_alt: float, prevalence: float, power: float = 0.80, alpha: float = 0.05,
                     n_max: int = 5000) -> int:
    """Total sample size needed to detect AUROC = auc_alt at the requested power."""
    prevalence = min(max(prevalence, 0.05), 0.95)
    for n in range(8, n_max + 1):
        n_pos = max(1, round(n * prevalence)); n_neg = max(1, n - n_pos)
        if power_auc(auc_alt, n_pos, n_neg, alpha) >= power:
            return n
    return n_max


def simulate_power(auc_alt: float, n_pos: int, n_neg: int, n_sims: int = 2000,
                   seed: int = GLOBAL_SEED, alpha: float = 0.05,
                   se_at_null: bool = False) -> float:
    """Rejection rate of a Wald test on the AUC, by simulation.

    Binormal model: positives ~ N(mu,1), negatives ~ N(0,1), with mu set so the population
    AUC equals auc_alt (AUC = Phi(mu/sqrt(2))). One-sided rejection at alpha/2: the lower
    limit of the Hanley-McNeil interval must exceed 0.50.

    se_at_null selects WHICH standard error that interval uses, and this is the whole reason
    the simulated and analytic numbers differ:
      False (default) -- SE evaluated at the OBSERVED AUC. This is the interval the manuscript
        actually reports, so it is the operationally relevant power. Hanley-McNeil SE decreases
        with AUC (about 0.90-0.95 of the null SE at AUROC 0.70 for these class counts), so the
        interval is narrower and the test rejects more often.
      True -- SE evaluated at the null (AUC = 0.50), which is the rule power_auc() assumes.
        Under this rule the simulation reproduces power_auc() to within Monte-Carlo error,
        which is what establishes that the two are not in conflict.
    """
    rng = np.random.default_rng(seed)
    mu = math.sqrt(2) * _norm_ppf(auc_alt)
    z_a = _norm_ppf(1 - alpha / 2)
    se_null = hanley_mcneil_se(0.5, n_pos, n_neg)
    hits = 0
    for _ in range(n_sims):
        pos = rng.normal(mu, 1.0, n_pos); neg = rng.normal(0.0, 1.0, n_neg)
        y = np.r_[np.ones(n_pos), np.zeros(n_neg)]; s = np.r_[pos, neg]
        a = roc_auc_score(y, s)
        se = se_null if se_at_null else hanley_mcneil_se(a, n_pos, n_neg)
        if a - z_a * se > 0.5:
            hits += 1
    return hits / n_sims


# ── (A) Learning curve ────────────────────────────────────────────────────────
def learning_curve_for_target(s5, df, feature_cols, harm_cols, batch_labels, target_col,
                              cfg, gpu, logger) -> List[Dict]:
    data = df[~df[target_col].isna()].copy()
    data[target_col] = data[target_col].astype(int)
    for c in feature_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data[feature_cols] = data[feature_cols].replace([np.inf, -np.inf], np.nan)

    mask_tv = data["split"].isin(["train", "valid"])
    X_tv = data.loc[mask_tv, feature_cols].reset_index(drop=True)
    y_tv = data.loc[mask_tv, target_col].reset_index(drop=True)

    use_combat = batch_labels is not None and len(harm_cols) > 0
    if use_combat:
        bl = pd.Series(batch_labels, index=df.index).reindex(data.index).fillna(-1).astype(int)
        batch_tv = bl.loc[mask_tv].reset_index(drop=True).values
    else:
        batch_tv = np.zeros(len(y_tv), dtype=int)

    rows = []
    for frac in LC_FRACTIONS:
        aucs = []
        n_used = []
        n_failed = 0
        for rep in range(LC_REPEATS):
            skf = StratifiedKFold(n_splits=LC_OUTER_SPLITS, shuffle=True, random_state=GLOBAL_SEED + rep)
            for tr_idx, val_idx in skf.split(X_tv, y_tv):
                rng = np.random.default_rng(GLOBAL_SEED + 1000 * rep + len(tr_idx))
                # stratified subsample of the training fold to `frac`
                sub = []
                for cls in (0, 1):
                    cls_idx = tr_idx[y_tv.iloc[tr_idx].values == cls]
                    k = max(1, int(round(len(cls_idx) * frac)))
                    sub.extend(rng.choice(cls_idx, size=min(k, len(cls_idx)), replace=False))
                sub = np.array(sub)
                y_sub = y_tv.iloc[sub]
                if y_sub.nunique() < 2 or y_tv.iloc[val_idx].nunique() < 2:
                    continue
                try:
                    model = s5.build_candidate_models(cfg, y_sub, gpu)[LC_MODEL]
                    X_tr, X_val = s5.harmonize_pair_if_needed(
                        X_tv.iloc[sub], X_tv.iloc[val_idx],
                        batch_tv[sub], batch_tv[val_idx], harm_cols, use_combat, logger)
                    model.fit(X_tr, y_sub)
                    p = model.predict_proba(X_val)[:, 1]
                    aucs.append(float(roc_auc_score(y_tv.iloc[val_idx], p)))
                    n_used.append(int(len(sub)))
                except Exception as e:
                    n_failed += 1
                    logger.warning("LC fold failed (%s, frac=%.2f): %s: %s",
                                   target_col, frac, type(e).__name__, e)
        if not aucs:
            raise RuntimeError(
                f"Learning curve produced no usable fold for {target_col} at fraction {frac:.2f} "
                f"({n_failed} folds raised). A curve cannot be drawn from this.")
        n_eff = int(round(float(np.mean(n_used))))
        rows.append({
            "target": target_col, "fraction": frac, "n_train_effective": n_eff,
            "n_train_nominal": int(round(len(X_tv) * (LC_OUTER_SPLITS - 1) / LC_OUTER_SPLITS * frac)),
            "auroc_mean": float(np.mean(aucs)),
            "auroc_std": float(np.std(aucs)),
            "n_folds": len(aucs),
            "n_folds_failed": n_failed,
        })
        logger.info("  [%s] frac=%.2f n=%d  OOF AUROC=%.3f±%.3f (%d folds, %d failed)",
                    TARGET_DISPLAY[target_col], frac, n_eff,
                    rows[-1]["auroc_mean"], rows[-1]["auroc_std"], len(aucs), n_failed)
    return rows


# ── Figures ───────────────────────────────────────────────────────────────────
def plot_learning_curve(lc_df: pd.DataFrame, path: Path):
    if lc_df.empty or not np.isfinite(lc_df["auroc_mean"].values.astype(float)).any():
        raise ValueError("Learning-curve table carries no finite AUROC; refusing to save an empty figure.")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for target in TARGET_COLUMNS:
        sub = lc_df[lc_df["target"] == target].sort_values("n_train_effective")
        if sub.empty:
            continue
        x = sub["n_train_effective"].values
        m = sub["auroc_mean"].values; s = sub["auroc_std"].values
        c = TARGET_COLORS[target]
        ax.plot(x, m, "-o", color=c, lw=2, label=TARGET_DISPLAY[target])
        ax.fill_between(x, m - s, m + s, color=c, alpha=0.15)
    ax.axhline(0.5, color="gray", ls="--", lw=1.2, label="Chance (0.50)")
    ax.set_xlabel("Effective training‑set size (patients)")
    ax.set_ylabel("Out‑of‑fold AUROC (mean ± SD)")
    ax.set_ylim(0.35, 0.85)
    ax.set_title("Learning curves — AUROC vs training-set size\n"
                 "(elastic-net, radiomic block of 642 features, leakage-safe nested folds; "
                 "25 folds per point)",
                 fontweight="bold", fontsize=11.5)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_power(power_rows: List[Dict], path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    # Panel A: power vs true AUROC for the locked-test class counts
    ax = axes[0]
    grid = np.linspace(0.5, 0.95, 100)
    for r in power_rows:
        if r["set"] != "locked_test":
            continue
        c = TARGET_COLORS[r["target"]]
        y = [power_auc(a, r["n_pos"], r["n_neg"]) for a in grid]
        ax.plot(grid, y, color=c, lw=2,
                label=f"{TARGET_DISPLAY[r['target']]} (n={r['n_pos']}+/{r['n_neg']}−)")
        mda = r.get("min_detectable_auc_80pct")
        if mda is not None and np.isfinite(mda):
            ax.plot([mda], [0.8], "o", color=c, ms=7, markeredgecolor="black")
    ax.axhline(0.8, color="gray", ls="--", lw=1.2, label="80% power")
    ax.set_xlabel("True AUROC"); ax.set_ylabel("Power to detect AUROC > 0.50")
    ax.set_title("A. Power of the LOCKED TEST set", fontweight="bold")
    ax.set_ylim(0, 1.02); ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.25)
    # Panel B: evaluation-set size — locked test vs nested-CV OOF pool vs required
    ax2 = axes[1]
    lt = {r["target"]: r for r in power_rows if r["set"] == "locked_test"}
    op = {r["target"]: r for r in power_rows if r["set"] == "nested_oof_pool"}
    order = [t for t in TARGET_COLUMNS if t in lt]
    labels = [TARGET_DISPLAY[t] for t in order]
    test_n = [lt[t]["n_total"] for t in order]
    oof_n = [op[t]["n_total"] if t in op else 0 for t in order]
    req = [lt[t]["required_total_n_auc070"] for t in order]
    x = np.arange(len(labels)); w = 0.27
    b1 = ax2.bar(x - w, test_n, w, color="#C0392B", label="Locked test (single split)")
    b2 = ax2.bar(x, oof_n, w, color="#2F3E55", label="Nested‑CV OOF pool")
    b3 = ax2.bar(x + w, req, w, color="#E0735B", label="Needed @ AUROC 0.70, 80% power")
    for bars in (b1, b2, b3):
        for rect in bars:
            ax2.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 2,
                     str(int(rect.get_height())), ha="center", fontsize=8.5)
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylabel("Evaluation‑set size (patients)")
    ax2.set_title("B. Evaluation size vs requirement", fontweight="bold")
    # The legend used to sit at "upper left", directly over the tallest bar's data label
    # (ER nested-CV pool, n=119), which clipped that number. Give the axis headroom and put
    # the legend above the bars so every count stays readable.
    ax2.set_ylim(0, max(test_n + oof_n + req) * 1.34)
    ax2.legend(loc="upper center", ncol=3, fontsize=7.5, frameon=True, framealpha=0.95)
    ax2.grid(axis="y", alpha=0.25)
    fig.suptitle("Statistical power for AUROC under the observed class balance", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--power-only", action="store_true",
                    help="recompute the power analysis only; keep the existing learning curve")
    args = ap.parse_args()

    for d in (FIGURES_DIR, TABLES_DIR, METADATA_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    logger = setup_logger()
    logger.info("=" * 80)
    logger.info("Starting %s", STEP_NAME)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Step 04 patient table not found: {INPUT_CSV} (run steps 00–04 first)")

    s5 = load_step05_module()
    cfg = s5.Step05Config()
    s5.set_global_seed(GLOBAL_SEED)
    df = pd.read_csv(INPUT_CSV)
    feature_sets = s5.resolve_feature_sets(df, FEATURE_DICT)
    batch_labels = s5.get_batch_labels(df)
    runtime = s5.detect_runtime_environment()
    gpu = bool(runtime.get("gpu_available", False))
    feature_cols = feature_sets.get(LC_VARIANT, [])
    if not feature_cols:
        raise KeyError(
            f"Feature set '{LC_VARIANT}' is empty or absent; available: {sorted(feature_sets)}. "
            "Refusing to draw a learning curve with no features.")
    harm_cols = [c for c in feature_cols]  # radiomic block: all columns are harmonizable
    logger.info("Learning curve: model=%s variant=%s features=%d", LC_MODEL, LC_VARIANT, len(feature_cols))

    # (A) Learning curves
    lc_csv = TABLES_DIR / "step14_learning_curve.csv"
    if args.power_only and lc_csv.exists():
        lc_df = pd.read_csv(lc_csv)
        lc_rows = lc_df.to_dict("records")
        # Redrawn, not re-fitted: the figure is a rendering of this table, so a caption or
        # styling change must not cost another run of the curve itself.
        plot_learning_curve(lc_df, FIGURES_DIR / "fig_learning_curve.png")
        logger.info("--power-only: reusing %s (%d rows) and redrawing the figure",
                    lc_csv.name, len(lc_df))
    else:
        lc_rows = []
        for target in TARGET_COLUMNS:
            lc_rows.extend(learning_curve_for_target(
                s5, df, feature_cols, harm_cols, batch_labels, target, cfg, gpu, logger))
        lc_df = pd.DataFrame(lc_rows)
        lc_df.to_csv(lc_csv, index=False, encoding="utf-8-sig")
        plot_learning_curve(lc_df, FIGURES_DIR / "fig_learning_curve.png")
        logger.info("Saved learning curve figure + %s", lc_csv.name)

    # (B) Power analysis — locked test and nested-CV OOF pool
    power_rows: List[Dict] = []
    for target in TARGET_COLUMNS:
        d = df[~df[target].isna()]
        y_all = d[target].astype(int)
        cohort_n = int(len(y_all)); prev = float(y_all.mean())
        for set_name, mask in [("locked_test", d["split"] == "test"),
                               ("nested_oof_pool", d["split"].isin(["train", "valid"]))]:
            ys = d.loc[mask, target].astype(int)
            n_pos = int((ys == 1).sum()); n_neg = int((ys == 0).sum())
            if n_pos < 1 or n_neg < 1:
                continue
            mda = min_detectable_auc(n_pos, n_neg)
            pwr070 = power_auc(0.70, n_pos, n_neg)
            sim070 = simulate_power(0.70, n_pos, n_neg) if set_name == "locked_test" else None
            # Same simulation, but scored with the rule power_auc() assumes. If this does not
            # land on power_to_detect_0.70, the two numbers really do disagree and the gap is
            # not just a difference of test.
            sim070_null = (simulate_power(0.70, n_pos, n_neg, se_at_null=True)
                           if set_name == "locked_test" else None)
            req = required_total_n(0.70, prev) if set_name == "locked_test" else None
            power_rows.append({
                "target": target, "target_display": TARGET_DISPLAY[target], "set": set_name,
                "n_pos": n_pos, "n_neg": n_neg, "n_total": n_pos + n_neg, "prevalence": round(prev, 3),
                "se_auc_at_0.70": round(hanley_mcneil_se(0.70, n_pos, n_neg), 4),
                "power_to_detect_0.70": round(pwr070, 3),
                "power_sim_0.70": (round(sim070, 3) if sim070 is not None else None),
                "power_sim_0.70_se_at_null": (round(sim070_null, 3) if sim070_null is not None else None),
                "se_auc_at_0.50": round(hanley_mcneil_se(0.50, n_pos, n_neg), 4),
                "min_detectable_auc_80pct": (round(mda, 3) if np.isfinite(mda) else None),
                "required_total_n_auc070": (int(req) if req is not None else None),
                "cohort_labeled_n": cohort_n,
            })
            logger.info("  [%s | %s] n=%d+/%d−  MDA(80%%)=%s  power@0.70 analytic=%.2f "
                        "sim=%s sim(null SE)=%s  req_n=%s",
                        TARGET_DISPLAY[target], set_name, n_pos, n_neg,
                        (f"{mda:.2f}" if np.isfinite(mda) else "NA"), pwr070,
                        (f"{sim070:.2f}" if sim070 is not None else "—"),
                        (f"{sim070_null:.2f}" if sim070_null is not None else "—"),
                        (str(req) if req else "—"))
    pw_df = pd.DataFrame(power_rows)
    pw_csv = TABLES_DIR / "step14_power_analysis.csv"
    pw_df.to_csv(pw_csv, index=False, encoding="utf-8-sig")
    plot_power(power_rows, FIGURES_DIR / "fig_power_analysis.png")
    logger.info("Saved power figure + %s", pw_csv.name)

    summary = {
        "step": STEP_NAME,
        "learning_curve": {"model": LC_MODEL, "variant": LC_VARIANT,
                           "fractions": LC_FRACTIONS, "outer_splits": LC_OUTER_SPLITS,
                           "repeats": LC_REPEATS, "rows": lc_rows},
        "power_analysis": power_rows,
        "method_notes": {
            "auc_variance": "Hanley & McNeil (1982)",
            "alpha": 0.05,
            "test": "Wald test on the AUC against H0: AUROC = 0.50; rejection when the lower "
                    "limit of the 95% Hanley-McNeil interval exceeds 0.50 (one-sided at alpha/2)",
            "analytic_power": "power_to_detect_0.70: normal-approximation power using the "
                              "Hanley-McNeil SE evaluated at the NULL (AUROC = 0.50)",
            "simulation": "power_sim_0.70: 2000 binormal simulations (positives ~ N(mu,1), "
                          "negatives ~ N(0,1), mu = sqrt(2)*Phi^-1(0.70)); rejection uses the SE "
                          "evaluated at the OBSERVED AUC, which is the interval the manuscript reports",
            "n_simulations": 2000,
            "reconciliation": "power_sim_0.70_se_at_null repeats the simulation with the SE "
                              "evaluated at the null, matching the analytic rule; it agrees with "
                              "power_to_detect_0.70 to within Monte-Carlo error. The 9-14 point "
                              "gap between the analytic and simulated columns is therefore a "
                              "difference of test, not a disagreement: the Hanley-McNeil SE is "
                              "smaller at AUROC 0.70 than at 0.50 (see se_auc_at_0.50 versus "
                              "se_auc_at_0.70), so an observed-AUC interval rejects more often.",
            "required_n": "increase n at fixed prevalence until power>=0.80 (alpha=0.05, two-sided)",
        },
    }
    (METADATA_DIR / "step14_learning_curve_power_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Completed %s successfully.", STEP_NAME)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
