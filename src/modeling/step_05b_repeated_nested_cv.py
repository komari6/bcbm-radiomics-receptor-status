"""
step_05b_repeated_nested_cv.py — repeated nested cross-validation for the tabular arm.

WHY THIS FILE EXISTS
--------------------
Step 05 reports one nested cross-validation (5 outer x 3 inner) and summarises it with a
t-interval over the five outer-fold AUROCs. Those five estimates share most of their training
data, so their variance is not the variance of an independent sample: there is no unbiased
estimator of k-fold cross-validation variance (Bengio & Grandvalet, JMLR 2004), and with small
n the resulting error bars are unreliable (Varoquaux, NeuroImage 2018).

This script answers that objection directly. It repeats the whole nested procedure R times with
different outer partitions and reports the distribution of *algorithm-level* performance across
repeats. The repeat-level estimates are far closer to exchangeable than folds within one
partition, so their spread is a defensible uncertainty statement — still conditional on this one
cohort, which the manuscript says explicitly.

The fold-internal pipeline is imported unchanged from step_05, so nothing about leakage control
differs between the two: the same ComBat-style harmonisation, missingness filter, imputation,
variance and correlation filters, stability selection, scaling and model pool, all fitted inside
the training fold only, and the same inner-fold model selection.

Usage:
    python src/modeling/step_05b_repeated_nested_cv.py                     # primary table
    python src/modeling/step_05b_repeated_nested_cv.py --input <csv> --tag onesession
    python src/modeling/step_05b_repeated_nested_cv.py --repeats 10 --jobs 6
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.modeling.step_05_tabular_modeling import (  # noqa: E402
    Step05Config,
    build_candidate_models,
    get_batch_labels,
    harmonize_pair_if_needed,
    resolve_feature_sets,
)

_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = _ROOT / "bcbm_project"
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = Path("./bcbm_project").resolve()
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "analysis_ready_step04_patient_level.csv"
FEATURE_DICT = PROJECT_ROOT / "metadata" / "step04_feature_dictionary.json"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
META_DIR = PROJECT_ROOT / "metadata"

TARGETS = ["ER_bin", "PR_bin", "HER2_bin"]
DISPLAY = {"ER_bin": "ER", "PR_bin": "PR", "HER2_bin": "HER2"}
# Feature blocks come from the step-04 dictionary so that "radiomics only" really is radiomics
# only: dropping the eight named scanner covariates still left field strength in the set under
# other names, which put the radiomics arm in competition with its own control.
VARIANTS = ["radiomics_pure", "radiomics_plus_burden", "all_features", "acquisition_only"]


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("step_05b")


def one_repeat(X_tv, y_tv, batch_tv, harm_cols, use_combat, cfg, n_outer, seed):
    """One complete outer partition: inner model selection per fold, then the outer estimate.

    Returns the mean of the outer-fold AUROCs and the AUROC of the pooled out-of-fold
    predictions for this partition. Both are reported: the first is the algorithm-level
    estimate, the second is what a single pooled ROC would give.
    """
    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    oof = np.full(len(y_tv), np.nan)
    fold_aucs, winners = [], []
    for fold_i, (tr, va) in enumerate(outer.split(X_tv, y_tv)):
        X_otr_raw, y_otr = X_tv.iloc[tr].copy(), y_tv.iloc[tr].copy()
        X_ova_raw, y_ova = X_tv.iloc[va].copy(), y_tv.iloc[va].copy()
        n_inner = max(min(cfg.inner_cv_splits, int(y_otr.value_counts().min()), len(y_otr)), 2)
        inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed + fold_i)
        scores: Dict[str, float] = {}
        for name, model in build_candidate_models(cfg, y_otr, False).items():
            aucs = []
            for itr, iva in inner.split(X_otr_raw, y_otr):
                y_itr, y_iva = y_otr.iloc[itr], y_otr.iloc[iva]
                if len(np.unique(y_itr)) < 2 or len(np.unique(y_iva)) < 2:
                    continue
                X_itr, X_iva = harmonize_pair_if_needed(
                    X_otr_raw.iloc[itr].copy(), X_otr_raw.iloc[iva].copy(),
                    batch_tv[tr][itr], batch_tv[tr][iva], harm_cols, use_combat, logging.getLogger("quiet"))
                try:
                    model.fit(X_itr, y_itr)
                    aucs.append(float(roc_auc_score(y_iva, model.predict_proba(X_iva)[:, 1])))
                except Exception:
                    pass
            scores[name] = float(np.mean(aucs)) if aucs else 0.0
        best = max(scores, key=scores.get) if scores else "elastic_net"
        X_otr, X_ova = harmonize_pair_if_needed(
            X_otr_raw, X_ova_raw, batch_tv[tr], batch_tv[va], harm_cols, use_combat, logging.getLogger("quiet"))
        try:
            m = build_candidate_models(cfg, y_otr, False)[best]
            m.fit(X_otr, y_otr)
            p = m.predict_proba(X_ova)[:, 1]
            oof[va] = p
            fold_aucs.append(float(roc_auc_score(y_ova, p)) if len(np.unique(y_ova)) > 1 else np.nan)
        except Exception:
            fold_aucs.append(np.nan)
        winners.append(best)
    ok = ~np.isnan(oof)
    pooled = float(roc_auc_score(y_tv[ok], oof[ok])) if ok.sum() and len(np.unique(y_tv[ok])) > 1 else np.nan
    return {"seed": seed, "mean_fold_auroc": float(np.nanmean(fold_aucs)),
            "pooled_oof_auroc": pooled, "winners": winners, "oof": oof}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--tag", default="primary")
    ap.add_argument("--feature-dict", default=str(FEATURE_DICT))
    ap.add_argument("--variants", default="",
                    help="comma-separated subset of VARIANTS; empty means all")
    a = ap.parse_args()
    wanted = [v.strip() for v in a.variants.split(",") if v.strip()] or list(VARIANTS)
    unknown = [v for v in wanted if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variant(s) {unknown}; choose from {VARIANTS}")
    log = setup_logger()

    cfg = Step05Config()
    df = pd.read_csv(a.input, low_memory=False)
    import json as _json
    fd = _json.load(open(a.feature_dict, encoding="utf-8"))
    rad, acq, bur = (fd["radiomic_block_columns"], fd["acquisition_block_columns"],
                     fd["burden_study_block_columns"])
    fs = {"radiomics_pure": rad,
          "radiomics_plus_burden": rad + bur,
          "all_features": rad + bur + acq,
          "acquisition_only": acq}
    fs = {k: [c for c in v if c in df.columns] for k, v in fs.items()}
    harm_cols = [c for c in rad if c in df.columns]
    batch_all = get_batch_labels(df)
    log.info("input=%s | rows=%d | repeats=%d | jobs=%d | feature sets=%s",
             Path(a.input).name, len(df), a.repeats, a.jobs, {k: len(v) for k, v in fs.items()})

    # One flat task list over (target, variant, repeat) so every worker stays busy: with a
    # per-combination pool the last wave of each combination leaves most cores idle.
    tasks, meta, ys, oofs, pids = [], [], {}, {}, {}
    rows: List[Dict] = []
    t0 = time.time()
    for target in TARGETS:
        for variant in wanted:
            cols = fs.get(variant, [])
            if not cols:
                continue
            d = df[~df[target].isna()].copy()
            d[target] = d[target].astype(int)
            for c in cols:
                d[c] = pd.to_numeric(d[c], errors="coerce")
            d[cols] = d[cols].replace([np.inf, -np.inf], np.nan)
            mask = d["split"].isin(["train", "valid"])
            X_tv = d.loc[mask, cols].reset_index(drop=True)
            y_tv = d.loc[mask, target].reset_index(drop=True)
            use_combat = batch_all is not None and variant != "acquisition_only" and len(harm_cols) > 0
            if use_combat:
                bl = pd.Series(batch_all, index=df.index).reindex(d.index).fillna(-1).astype(int)
                batch_tv = bl.loc[mask].values
            else:
                batch_tv = np.zeros(len(y_tv), dtype=int)
            n_outer = max(min(cfg.outer_cv_splits, int(y_tv.value_counts().min())), 2)
            ys[(target, variant)] = y_tv.values
            pid_col = next((c for c in ("patient_base", "patient_id") if c in d.columns), None)
            pids[(target, variant)] = (d.loc[mask, pid_col].reset_index(drop=True).astype(str).values
                                       if pid_col else np.arange(int(mask.sum())).astype(str))
            for r in range(a.repeats):
                tasks.append(delayed(one_repeat)(X_tv, y_tv, batch_tv, harm_cols, use_combat, cfg,
                                                 n_outer, cfg.global_seed + 1000 * r))
                meta.append((target, variant, r, int(len(y_tv)), len(cols)))

    log.info("running %d tasks (%d combinations x %d repeats) on %d workers",
             len(tasks), len(tasks) // max(a.repeats, 1), a.repeats, a.jobs)
    res = Parallel(n_jobs=a.jobs, backend="loky", inner_max_num_threads=2, verbose=5)(tasks)
    for (target, variant, r, n_tv, n_feat), out in zip(meta, res):
        rows.append({"target": target, "target_display": DISPLAY[target], "feature_set_variant": variant,
                     "repeat": r, "n_trainval": n_tv, "n_features": n_feat,
                     "mean_fold_auroc": out["mean_fold_auroc"], "pooled_oof_auroc": out["pooled_oof_auroc"],
                     "modal_winner": max(set(out["winners"]), key=out["winners"].count)})
    for (target, variant, r, n_tv, n_feat), out in zip(meta, res):
        oofs.setdefault((target, variant), []).append(out["oof"])
    log.info("all tasks finished in %.1f min", (time.time() - t0) / 60)

    # Patient-level bootstrap on the cross-fitted scores. Averaging each patient's out-of-fold
    # probability over the repetitions gives one score per patient; resampling patients then gives
    # an interval that reflects who is in the cohort, which the partition spread does not. It is
    # conditional on the fitted fold models, and the manuscript says so.
    rng = np.random.default_rng(cfg.global_seed)
    boot_rows = []
    for (target, variant), mats in oofs.items():
        y = ys[(target, variant)]
        p_mean = np.nanmean(np.vstack(mats), axis=0)
        ok = ~np.isnan(p_mean)
        if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
            continue
        yv, pv = y[ok], p_mean[ok]
        point = float(roc_auc_score(yv, pv))
        bs = []
        for _ in range(2000):
            idx = rng.integers(0, len(yv), len(yv))
            if len(np.unique(yv[idx])) < 2:
                continue
            bs.append(roc_auc_score(yv[idx], pv[idx]))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        boot_rows.append({"target": target, "target_display": DISPLAY[target],
                          "feature_set_variant": variant, "n_patients": int(ok.sum()),
                          "cross_fitted_auroc": point, "boot_ci_lo": float(lo), "boot_ci_hi": float(hi),
                          "n_bootstrap": len(bs)})
        log.info("  bootstrap %-5s %-22s AUROC %.3f  95%% CI %.3f-%.3f",
                 DISPLAY[target], variant, point, lo, hi)
    pd.DataFrame(boot_rows).to_csv(TABLES_DIR / f"step05b_patient_bootstrap_{a.tag}.csv",
                                   index=False, encoding="utf-8-sig")

    # The scores the bootstrap above is computed on. Without these, ROC and PR curves can only
    # be reconstructed parametrically from a single (AUROC, sensitivity, specificity) triple,
    # which draws a curve shape that was never observed.
    score_rows = []
    for (target, variant), mats in oofs.items():
        y = ys[(target, variant)]
        p_mean = np.nanmean(np.vstack(mats), axis=0)
        pid = pids.get((target, variant), np.arange(len(y)).astype(str))
        for i in range(len(y)):
            if np.isnan(p_mean[i]):
                continue
            score_rows.append({"target": target, "target_display": DISPLAY[target],
                               "feature_set_variant": variant, "patient": pid[i],
                               "y_true": int(y[i]), "score": float(p_mean[i]),
                               "n_repeats_averaged": int(np.sum(~np.isnan(np.vstack(mats)[:, i])))})
    pd.DataFrame(score_rows).to_csv(TABLES_DIR / f"step05b_crossfitted_scores_{a.tag}.csv",
                                    index=False, encoding="utf-8-sig")
    log.info("wrote %d cross-fitted patient scores", len(score_rows))

    per = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    per.to_csv(TABLES_DIR / f"step05b_repeated_nested_cv_{a.tag}_per_repeat.csv", index=False, encoding="utf-8-sig")
    g = per.groupby(["target_display", "feature_set_variant"])["mean_fold_auroc"]
    summ = pd.DataFrame({"mean": g.mean(), "sd": g.std(ddof=1),
                         "pct2_5": g.quantile(0.025), "pct97_5": g.quantile(0.975),
                         "min": g.min(), "max": g.max(), "n_repeats": g.count()}).reset_index()
    summ.to_csv(TABLES_DIR / f"step05b_repeated_nested_cv_{a.tag}_summary.csv", index=False, encoding="utf-8-sig")
    META_DIR.mkdir(parents=True, exist_ok=True)
    json.dump({"input": str(a.input), "repeats": a.repeats, "outer_splits": cfg.outer_cv_splits,
               "inner_splits": cfg.inner_cv_splits, "seed_base": cfg.global_seed,
               "runtime_minutes": round((time.time() - t0) / 60, 1)},
              open(META_DIR / f"step05b_repeated_nested_cv_{a.tag}.json", "w", encoding="utf-8"), indent=1)
    log.info("done in %.1f min -> %s", (time.time() - t0) / 60, TABLES_DIR)
    print(summ.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
