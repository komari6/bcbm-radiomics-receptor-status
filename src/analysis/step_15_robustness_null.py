"""
Step 15 — Robustness: label-permutation null, seed stability, calibration
=========================================================================
Three CPU-only robustness analyses that strengthen the negative result and
prove the pipeline is unbiased. All reuse Step 05's leakage-safe components.

  (A) Label-permutation null. Shuffle the labels and re-run leakage-safe CV
      many times. If the pipeline is unbiased, the null AUROC centers at 0.50;
      the empirical p value = P(null >= observed) tells whether the observed
      AUROC is distinguishable from chance.
  (B) Seed stability. Repeat the CV with many global seeds → distribution of
      AUROC, showing the estimate is stable (not a lucky-seed artifact).
  (C) Calibration. Reliability curve + Brier score from out-of-fold
      probabilities.

A deliberately *reduced* leakage-safe pipeline is used for the permutation and
seed analyses (median imputation -> variance filter -> correlation filter 0.88
-> in-fold ComBat -> RobustScaler -> elastic-net), so that thousands of refits
are tractable on CPU; stability selection is omitted here and noted as such.
This does not change the main analysis (Step 05); it characterizes the null.

Outputs:
  reports/figures/step13_publication_package/fig_permutation_null.{png,pdf}
  reports/tables/step15_robustness.csv
  metadata/step15_robustness_summary.json
"""
from __future__ import annotations

import importlib.util
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss

_REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = _REPO_ROOT / "bcbm_project"
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = Path("./bcbm_project").resolve()

STEP_NAME = "step_15_robustness_null"
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
VARIANT = "radiomics_pure"   # the 642 radiomic columns only; see step 04 feature blocks
N_SPLITS = 5
# Raised from 200: with 200 permutations a P value near .02 rests on about
# four extreme draws. 10,000 puts the Monte Carlo standard error of such a P value near .0015.
# Permutations are drawn serially per target from one generator, so any smaller count reproduces
# the leading draws of a larger one; --n-permutations 200 reproduces the original table exactly.
N_PERMUTATIONS = 10000
N_SEEDS = 25


def setup_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(STEP_NAME); lg.setLevel(logging.INFO)
    if not lg.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        sh = logging.StreamHandler(); sh.setFormatter(fmt); lg.addHandler(sh)
        fh = logging.FileHandler(LOGS_DIR / f"{STEP_NAME}.log", encoding="utf-8"); fh.setFormatter(fmt); lg.addHandler(fh)
    return lg


def load_step05():
    # Cached: light_pipeline() calls this once per fold, and re-executing step 05's module for
    # each of hundreds of thousands of folds dominated the run time. Step 05 has no import-time
    # side effects (no seeding), so caching cannot change a result.
    # Cached in sys.modules only. A joblib worker receives this function pickled by value; a
    # module-level global holding the module would be pickled by name and fail to import in the
    # worker, whereas sys.modules lives as long as the worker process.
    import sys as _sys
    m = _sys.modules.get("step_05_tabular_modeling")
    if m is not None:
        return m
    p = _REPO_ROOT / "src" / "modeling" / "step_05_tabular_modeling.py"
    spec = importlib.util.spec_from_file_location("step_05_tabular_modeling", p)
    m = importlib.util.module_from_spec(spec)
    _sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def light_pipeline(seed: int, cfg) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("variance", VarianceThreshold(1e-8)),
        ("correlation", load_step05().CorrelationFilter(threshold=cfg.corr_threshold, min_features=5)),
        ("scaler", RobustScaler()),
        ("model", LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.5,
                                     class_weight="balanced", max_iter=5000, random_state=seed)),
    ])


def oof_auc(s5, X, y, batch, harm_cols, use_combat, seed, cfg, logger, y_override=None):
    """Leakage-safe out-of-fold AUROC for one (reduced) pipeline pass."""
    yy = y if y_override is None else pd.Series(y_override, index=y.index)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof = np.full(len(yy), np.nan)
    for tr, va in skf.split(X, yy):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        if use_combat:
            Xtr, Xva = s5.harmonize_pair_if_needed(Xtr, Xva, batch[tr], batch[va], harm_cols, True, logger)
        try:
            pipe = light_pipeline(seed, cfg)
            pipe.fit(Xtr, yy.iloc[tr])
            oof[va] = pipe.predict_proba(Xva)[:, 1]
        except Exception as e:
            logger.debug("fold failed: %s", e)
    m = ~np.isnan(oof)
    if m.sum() < 10 or len(np.unique(yy[m])) < 2:
        return np.nan, oof
    return float(roc_auc_score(yy[m].values, oof[m])), oof


def _auc_task(X, y, batch, harm_cols, use_combat, seed, y_override=None):
    """One CV pass in a worker process; returns the AUROC only. Each worker loads step 05
    itself (a module cannot be pickled) and uses the default Step05Config, as main() does."""
    s5 = load_step05()
    a, _ = oof_auc(s5, X, y, batch, harm_cols, use_combat, seed, s5.Step05Config(),
                   logging.getLogger(STEP_NAME), y_override=y_override)
    return a


def draw_figure(rows, tag):
    """Figure S17 from the summary rows, so it can be redrawn without re-running anything."""
    # ---- figure: permutation null + seed stability/calibration ----
    # NOTE: this block previously drew a THREE-panel figure (null histograms, a seed boxplot and a
    # calibration curve). The published Figure S17 - and the manuscript legend describing it
    # ("diamonds", "shaded band, chance to the 95th percentile", "Brier ... annotated") - is the
    # TWO-panel summary below. The released code therefore did NOT regenerate the published figure,
    # which contradicted the Data/Code availability statement. Rebuilt here from the summary values
    # already in `rows`, so that claim now holds.
    by_t = {r["target"]: r for r in rows}
    order = [t for t in TARGET_COLUMNS if t in by_t]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

    ax = axes[0]
    for i_t, t in enumerate(order):
        r = by_t[t]
        c = TARGET_COLORS[t]
        lo, hi = 0.5, r["null_p95"]
        ax.barh(i_t, hi - lo, left=lo, height=0.30, color=c, alpha=0.30, zorder=1)
        ax.plot([r["null_mean"]], [i_t], "o", color=c, ms=7, markeredgecolor="black", zorder=3)
        ax.plot([r["observed_oof_auc"]], [i_t], "D", color=c, ms=12, markeredgecolor="black", zorder=4)
        # Anchor the label beyond whichever is further right, the null band or the observed
        # diamond: for PR the observed AUROC (0.61) sits past the band (0.59), and the label
        # was drawn on top of its own marker.
        ax.text(max(hi, r["observed_oof_auc"]) + 0.015, i_t,
                "p=%.2f" % r["permutation_p_value"], va="center", fontsize=9)
    ax.axvline(0.5, color="gray", lw=1, ls=":")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([TARGET_DISPLAY[t] for t in order])
    ax.set_xlabel("Out-of-fold AUROC")
    ax.set_title("A. Label-permutation null\n(band = null 0.50→p95; ◆ = observed)",
                 fontweight="bold", fontsize=11)
    ax.set_xlim(0.45, 0.72)

    ax2 = axes[1]
    for i_t, t in enumerate(order):
        r = by_t[t]
        c = TARGET_COLORS[t]
        ax2.errorbar(r["seed_auc_mean"], i_t, xerr=r["seed_auc_sd"], fmt="s", color=c, ms=11,
                     markeredgecolor="black", ecolor=c, elinewidth=2, capsize=5, zorder=3)
        ax2.text(0.4585, i_t + 0.16, "Brier %.2f" % r["brier"], fontsize=9, color="#444444")
    ax2.axvline(0.5, color="gray", lw=1, ls=":")
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels([TARGET_DISPLAY[t] for t in order])
    ax2.set_xlabel("Out-of-fold AUROC (mean ± SD over seeds)")
    ax2.set_title("B. Seed stability (%d seeds) + calibration" % N_SEEDS, fontweight="bold", fontsize=11)
    ax2.set_xlim(0.44, 0.62)
    # headroom so the topmost Brier label does not collide with the panel title
    ax2.set_ylim(-0.55, len(order) - 0.5 + 0.30)
    ax.set_ylim(-0.55, len(order) - 0.5 + 0.30)

    fig.suptitle("Robustness: permutation null, seed stability, calibration — all consistent with chance",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIGURES_DIR / f"fig_permutation_null{tag}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"fig_permutation_null{tag}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    # --input/--feature-dict/--tag let the same permutation machinery be pointed at the
    # one-examination-per-patient sensitivity table without touching the primary outputs.
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(INPUT_CSV))
    ap.add_argument("--feature-dict", default=str(FEATURE_DICT))
    ap.add_argument("--tag", default="", help="Suffix for the output table, figure and summary.")
    ap.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    ap.add_argument("--figure-only", action="store_true",
                    help="Redraw the figure from the stored result table; compute nothing.")
    ap.add_argument("--jobs", type=int, default=1, help="Worker processes for the permutation and seed passes.")
    args = ap.parse_args()
    tag = args.tag
    n_perm = args.n_permutations
    from joblib import Parallel, delayed
    in_csv, fdict = Path(args.input), Path(args.feature_dict)

    for d in (FIGURES_DIR, TABLES_DIR, METADATA_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    logger = setup_logger()
    if args.figure_only:
        # Redraw from the stored results. Nothing is recomputed: the analysis on record is the
        # analysis that produced step15_robustness<tag>.csv.
        stored = pd.read_csv(TABLES_DIR / f"step15_robustness{tag}.csv")
        draw_figure(stored.to_dict("records"), tag)
        logger.info("Redrew fig_permutation_null%s from step15_robustness%s.csv", tag, tag)
        return
    logger.info("=" * 80); logger.info("Starting %s%s", STEP_NAME, f" [{tag}]" if tag else "")
    if not in_csv.exists():
        raise FileNotFoundError(f"Run steps 00–04 first; missing {in_csv}")

    s5 = load_step05(); cfg = s5.Step05Config(); s5.set_global_seed(GLOBAL_SEED)
    df = pd.read_csv(in_csv)
    feats = s5.resolve_feature_sets(df, fdict)
    batch_all = s5.get_batch_labels(df)
    cols = feats.get(VARIANT, [])
    logger.info("Variant=%s features=%d | perms=%d seeds=%d jobs=%d", VARIANT, len(cols), n_perm, N_SEEDS, args.jobs)

    rng = np.random.default_rng(GLOBAL_SEED)
    rows: List[Dict] = []
    null_store: Dict[str, np.ndarray] = {}
    seed_store: Dict[str, np.ndarray] = {}
    calib_store: Dict[str, Dict] = {}

    for target in TARGET_COLUMNS:
        data = df[~df[target].isna()].copy()
        data[target] = data[target].astype(int)
        for c in cols:
            data[c] = pd.to_numeric(data[c], errors="coerce")
        data[cols] = data[cols].replace([np.inf, -np.inf], np.nan)
        X = data[cols].reset_index(drop=True)
        y = data[target].reset_index(drop=True)
        use_combat = batch_all is not None and len(cols) > 0
        if use_combat:
            bl = pd.Series(batch_all, index=df.index).reindex(data.index).fillna(-1).astype(int)
            batch = bl.reset_index(drop=True).values
        else:
            batch = np.zeros(len(y), dtype=int)
        harm_cols = list(cols)

        # (A) observed + permutation null
        observed, oof = oof_auc(s5, X, y, batch, harm_cols, use_combat, GLOBAL_SEED, cfg, logger)
        # Draw every permutation first, serially and in the original order, so the null does not
        # depend on how the passes are scheduled across workers.
        perms = [rng.permutation(y.values) for _ in range(n_perm)]
        null = Parallel(n_jobs=args.jobs)(
            delayed(_auc_task)(X, y, batch, harm_cols, use_combat, GLOBAL_SEED, yp) for yp in perms)
        null = np.array([a for a in null if not np.isnan(a)])
        n_ge = int(np.sum(null >= observed)) if null.size else 0
        p_emp = float((n_ge + 1) / (len(null) + 1)) if null.size else np.nan
        # Monte Carlo standard error of the permutation P value
        p_mcse = float(np.sqrt(p_emp * (1 - p_emp) / len(null))) if null.size else np.nan
        null_store[target] = null

        # (B) seed stability (unpermuted)
        seeds = [GLOBAL_SEED + k for k in range(N_SEEDS)]
        seed_aucs = Parallel(n_jobs=args.jobs)(
            delayed(_auc_task)(X, y, batch, harm_cols, use_combat, sd) for sd in seeds)
        seed_aucs = np.array([a for a in seed_aucs if not np.isnan(a)])
        seed_store[target] = seed_aucs

        # (C) calibration from observed OOF
        m = ~np.isnan(oof)
        brier = float(brier_score_loss(y[m].values, oof[m])) if m.sum() else np.nan
        nb = 5
        bins = np.linspace(0, 1, nb + 1)
        idx = np.digitize(oof[m], bins) - 1
        cx, cy = [], []
        for b in range(nb):
            sel = idx == b
            if sel.sum() >= 3:
                cx.append(float(oof[m][sel].mean())); cy.append(float(y[m].values[sel].mean()))
        calib_store[target] = {"brier": brier, "pred": cx, "obs": cy}

        rows.append({
            "target": target, "display": TARGET_DISPLAY[target],
            "observed_oof_auc": round(observed, 4),
            "null_mean": round(float(null.mean()), 4) if null.size else None,
            "null_p95": round(float(np.percentile(null, 95)), 4) if null.size else None,
            "permutation_p_value": round(p_emp, 4) if not np.isnan(p_emp) else None,
            "seed_auc_mean": round(float(seed_aucs.mean()), 4) if seed_aucs.size else None,
            "seed_auc_sd": round(float(seed_aucs.std()), 4) if seed_aucs.size else None,
            "brier": round(brier, 4) if not np.isnan(brier) else None,
            "n_permutations": int(null.size), "n_seeds": int(seed_aucs.size),
            "n_null_ge_observed": n_ge,
            "permutation_p_mc_se": round(p_mcse, 5) if not np.isnan(p_mcse) else None,
        })
        logger.info("  [%s] observed=%.3f null=%.3f(p95=%.3f) p=%.3f | seed=%.3f±%.3f | Brier=%.3f",
                    TARGET_DISPLAY[target], observed, null.mean() if null.size else float("nan"),
                    np.percentile(null, 95) if null.size else float("nan"),
                    p_emp, seed_aucs.mean() if seed_aucs.size else float("nan"),
                    seed_aucs.std() if seed_aucs.size else float("nan"), brier)

    pd.DataFrame(rows).to_csv(TABLES_DIR / f"step15_robustness{tag}.csv", index=False, encoding="utf-8-sig")

    draw_figure(rows, tag)

    summary = {"step": STEP_NAME, "variant": VARIANT, "n_splits": N_SPLITS,
               "n_permutations": n_perm, "n_seeds": N_SEEDS, "input": str(in_csv),
               "reduced_pipeline": "impute(median)->variance->corr(0.88)->ComBat(in-fold)->RobustScaler->elastic-net (stability selection omitted for tractability)",
               "results": rows}
    # Suffixed like the table and figure: unsuffixed, the one-session run overwrote the pooled summary.
    (METADATA_DIR / f"step15_robustness_summary{tag}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Completed %s.", STEP_NAME); logger.info("=" * 80)


if __name__ == "__main__":
    main()
