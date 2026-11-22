"""
step_19_unit_of_analysis.py — random versus patient-grouped splitting at the lesion and case level
(analyses A1 and A2 of the protocol in docs/PROTOCOL_added_analyses.md).

WHY THIS FILE EXISTS
--------------------
Two questions about the same data:

A1  How much does splitting by lesion or by image (case), instead of by patient, inflate AUROC
    when nothing else changes? Earlier work on this cohort treated lesions as independent
    samples, so lesions of one patient could sit on both sides of a split.
A2  Does keeping lesions (or cases) as separate training rows, with every patient confined to one
    side of each split, recover signal that patient-level aggregation discards?

Both are answered by one nested cross-validation run over four arms:

    L-rand   lesion rows, StratifiedKFold (outer and inner)
    L-group  lesion rows, StratifiedGroupKFold by patient (outer and inner)
    C-rand   case rows,   StratifiedKFold
    C-group  case rows,   StratifiedGroupKFold by patient

Within each pair the only difference is the splitter. The model pool, fold-internal
preprocessing, ComBat harmonisation (batch = the row's own field strength), stability selection,
inner-fold model selection, fold counts and seeds are those of step 05 / step 05b, imported
unchanged. The pool is the 123 train + valid patients of the primary analysis; the locked-test
patients stay out, as in Table 2.

A2 is read from the grouped arms: the out-of-fold row probabilities are averaged per patient
(primary rule, fixed in the protocol), with the maximum and the largest-lesion probability as
secondary rules.

A secondary "Taha-like" arm fits an untuned Random Forest on lesion rows with a single 80:20
split, random or grouped by patient, over many seeds. It illustrates that style of evaluation; it
is not a reproduction of the published run, which cannot be repeated from the released data.

Usage:
    python src/analysis/step_19_unit_of_analysis.py --repeats 1 --rf-seeds 5 --tag unitdemo_trial
    python src/analysis/step_19_unit_of_analysis.py --repeats 10 --rf-seeds 100 --jobs 10 --tag unitdemo
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold,
                                     train_test_split)
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.modeling.step_05_tabular_modeling import (  # noqa: E402
    Step05Config,
    build_candidate_models,
    harmonize_pair_if_needed,
)

_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = _ROOT / "bcbm_project"
LESION_TABLE = PROJECT_ROOT / "data" / "processed" / "analysis_ready_step03_lesion_only.csv"
CASE_TABLE = PROJECT_ROOT / "data" / "processed" / "analysis_ready_step04_case_level.csv"
SPLITS = PROJECT_ROOT / "metadata" / "step04_patient_splits.csv"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
META_DIR = PROJECT_ROOT / "metadata"

TARGETS = ["ER_bin", "PR_bin", "HER2_bin"]
DISPLAY = {"ER_bin": "ER", "PR_bin": "PR", "HER2_bin": "HER2"}
ARMS = ["L-rand", "L-group", "C-rand", "C-group"]
BATCH_COL = "dominant_field_strength_t"   # the column step 05's get_batch_labels resolves to


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("step_19")


# ----------------------------------------------------------------------------- data

def load_units(log: logging.Logger) -> Dict[str, Dict]:
    """Lesion and case tables restricted to the train + valid patients of the primary split."""
    splits = pd.read_csv(SPLITS)
    tv = set(splits.loc[splits["split"].isin(["train", "valid"]), "patient_base"].astype(str))

    les = pd.read_csv(LESION_TABLE, low_memory=False)
    les["patient_base"] = les["patient_base"].astype(str)
    les = les[les["patient_base"].isin(tv)].reset_index(drop=True)
    les[BATCH_COL] = pd.to_numeric(les["Magnetic_Field_Strength_ID"], errors="coerce")
    les["row_id"] = les["case_id"].astype(str) + "|" + les["mask_filename"].astype(str)
    les["size_for_largest"] = pd.to_numeric(les["mask_volume_cc"], errors="coerce")
    les_feats = [c for c in les.columns if c.startswith("original_")]

    cas = pd.read_csv(CASE_TABLE, low_memory=False)
    cas["patient_base"] = cas["patient_base"].astype(str)
    cas = cas[cas["patient_base"].isin(tv)].reset_index(drop=True)
    cas["row_id"] = cas["case_id"].astype(str)
    cas["size_for_largest"] = pd.to_numeric(cas["case_total_volume_cc"], errors="coerce")
    cas_feats = [c for c in cas.columns if c.startswith("case_original_")]

    log.info("pool: %d train+valid patients | lesion rows %d (%d patients, %d features) | "
             "case rows %d (%d patients, %d features)", len(tv), len(les), les["patient_base"].nunique(),
             len(les_feats), len(cas), cas["patient_base"].nunique(), len(cas_feats))
    return {"L": {"df": les, "feats": les_feats}, "C": {"df": cas, "feats": cas_feats}}


def prepare(unit: Dict, target: str):
    d = unit["df"][~unit["df"][target].isna()].copy().reset_index(drop=True)
    cols = unit["feats"]
    X = d[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    y = d[target].astype(int)
    groups = d["patient_base"].values
    batch = pd.to_numeric(d[BATCH_COL], errors="coerce").fillna(-1).astype(int).values
    return d, X, y, groups, batch, cols


def make_splitter(grouped: bool, n_splits: int, seed: int):
    if grouped:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


# ----------------------------------------------------------------------------- nested CV

def one_repeat(X, y, groups, batch, harm_cols, grouped, cfg, seed):
    """One outer partition with inner model selection, as step 05b's one_repeat, with the
    splitter (and its groups) as the only free choice."""
    quiet = logging.getLogger("quiet")
    outer = make_splitter(grouped, cfg.outer_cv_splits, seed)
    oof = np.full(len(y), np.nan)
    fold_of = np.full(len(y), -1)
    fold_aucs, winners = [], []
    for fold_i, (tr, va) in enumerate(outer.split(X, y, groups if grouped else None)):
        fold_of[va] = fold_i
        X_otr_raw, y_otr = X.iloc[tr].copy(), y.iloc[tr].copy()
        X_ova_raw, y_ova = X.iloc[va].copy(), y.iloc[va].copy()
        n_inner = max(min(cfg.inner_cv_splits, int(y_otr.value_counts().min())), 2)
        inner = make_splitter(grouped, n_inner, seed + fold_i)
        scores: Dict[str, float] = {}
        for name, model in build_candidate_models(cfg, y_otr, False).items():
            aucs = []
            for itr, iva in inner.split(X_otr_raw, y_otr, groups[tr] if grouped else None):
                y_itr, y_iva = y_otr.iloc[itr], y_otr.iloc[iva]
                if len(np.unique(y_itr)) < 2 or len(np.unique(y_iva)) < 2:
                    continue
                X_itr, X_iva = harmonize_pair_if_needed(
                    X_otr_raw.iloc[itr].copy(), X_otr_raw.iloc[iva].copy(),
                    batch[tr][itr], batch[tr][iva], harm_cols, True, quiet)
                try:
                    model.fit(X_itr, y_itr)
                    aucs.append(float(roc_auc_score(y_iva, model.predict_proba(X_iva)[:, 1])))
                except Exception:
                    pass
            scores[name] = float(np.mean(aucs)) if aucs else 0.0
        best = max(scores, key=scores.get) if scores else "elastic_net"
        X_otr, X_ova = harmonize_pair_if_needed(X_otr_raw, X_ova_raw, batch[tr], batch[va],
                                                harm_cols, True, quiet)
        try:
            m = build_candidate_models(cfg, y_otr, False)[best]
            m.fit(X_otr, y_otr)
            p = m.predict_proba(X_ova)[:, 1]
            oof[va] = p
            fold_aucs.append(float(roc_auc_score(y_ova, p)) if len(np.unique(y_ova)) > 1 else np.nan)
        except Exception:
            fold_aucs.append(np.nan)
        winners.append(best)
    return {"oof": oof, "fold": fold_of, "mean_fold_auroc": float(np.nanmean(fold_aucs)),
            "winners": winners}


def patient_scores(d: pd.DataFrame, target: str, oof: np.ndarray) -> pd.DataFrame:
    """Aggregate out-of-fold row probabilities to one score per patient under the three rules."""
    t = pd.DataFrame({"patient": d["patient_base"].values, "y": d[target].astype(int).values,
                      "p": oof, "size": d["size_for_largest"].values}).dropna(subset=["p"])
    out = []
    for pid, g in t.groupby("patient"):
        big = g.loc[g["size"].idxmax(), "p"] if g["size"].notna().any() else g["p"].iloc[0]
        out.append({"patient": pid, "y": int(g["y"].iloc[0]), "n_rows": len(g),
                    "mean": g["p"].mean(), "max": g["p"].max(), "largest": big})
    return pd.DataFrame(out)


def safe_auc(y, p) -> float:
    y, p = np.asarray(y), np.asarray(p)
    ok = ~np.isnan(p)
    return float(roc_auc_score(y[ok], p[ok])) if ok.sum() and len(np.unique(y[ok])) > 1 else np.nan


# ----------------------------------------------------------------------------- Taha-like arm

def rf_once(X, y, groups, grouped, seed):
    if grouped:
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(X, y, groups))
    else:
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
    if len(np.unique(y.iloc[te])) < 2:
        return np.nan
    rf = Pipeline([("impute", SimpleImputer(strategy="median")),
                   ("model", RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=1))])
    rf.fit(X.iloc[tr], y.iloc[tr])
    return float(roc_auc_score(y.iloc[te], rf.predict_proba(X.iloc[te])[:, 1]))


# ----------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--rf-seeds", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--tag", default="unitdemo")
    ap.add_argument("--arms", default=",".join(ARMS))
    a = ap.parse_args()
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    log = setup_logger()
    cfg = Step05Config()
    units = load_units(log)
    t0 = time.time()

    prepared, tasks, meta = {}, [], []
    for arm in arms:
        unit_key, grouped = arm[0], arm.endswith("group")
        for target in TARGETS:
            d, X, y, groups, batch, cols = prepare(units[unit_key], target)
            prepared[(arm, target)] = d
            for r in range(a.repeats):
                seed = cfg.global_seed + 1000 * r
                tasks.append(delayed(one_repeat)(X, y, groups, batch, cols, grouped, cfg, seed))
                meta.append((arm, target, r, seed, len(y), d["patient_base"].nunique(), len(cols)))
    log.info("nested CV: %d tasks (%d arms x %d targets x %d repeats) on %d workers",
             len(tasks), len(arms), len(TARGETS), a.repeats, a.jobs)
    res = Parallel(n_jobs=a.jobs, backend="loky", inner_max_num_threads=2, verbose=5)(tasks)
    log.info("nested CV finished in %.1f min", (time.time() - t0) / 60)

    per_rows, oof_rows, pat_store = [], [], {}
    for (arm, target, r, seed, n_rows, n_pat, n_feat), out in zip(meta, res):
        d = prepared[(arm, target)]
        ps = patient_scores(d, target, out["oof"])
        pat_store.setdefault((arm, target), []).append(ps.set_index("patient"))
        per_rows.append({"arm": arm, "unit": "lesion" if arm[0] == "L" else "case",
                         "splitter": "grouped" if arm.endswith("group") else "random",
                         "target": DISPLAY[target], "repeat": r, "seed": seed,
                         "n_rows": n_rows, "n_patients": n_pat, "n_features": n_feat,
                         "mean_fold_auroc": out["mean_fold_auroc"],
                         "pooled_oof_auroc_rows": safe_auc(d[target].astype(int), out["oof"]),
                         "patient_auroc_mean": safe_auc(ps["y"], ps["mean"]),
                         "patient_auroc_max": safe_auc(ps["y"], ps["max"]),
                         "patient_auroc_largest": safe_auc(ps["y"], ps["largest"]),
                         "modal_winner": max(set(out["winners"]), key=out["winners"].count)})
        oof_rows.append(pd.DataFrame({"arm": arm, "target": DISPLAY[target], "repeat": r, "seed": seed,
                                      "row_id": d["row_id"].values, "patient": d["patient_base"].values,
                                      "fold": out["fold"], "y_true": d[target].astype(int).values,
                                      "oof_prob": out["oof"]}))
    per = pd.DataFrame(per_rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    per.to_csv(TABLES_DIR / f"step19_{a.tag}_per_repeat.csv", index=False, encoding="utf-8-sig")
    pd.concat(oof_rows).to_csv(TABLES_DIR / f"step19_{a.tag}_oof_and_partitions.csv", index=False,
                               encoding="utf-8-sig")

    # Summary across repeats, per arm and receptor.
    summ_rows = []
    for (arm, tgt), g in per.groupby(["arm", "target"]):
        for metric in ["mean_fold_auroc", "patient_auroc_mean", "patient_auroc_max", "patient_auroc_largest"]:
            v = g[metric].dropna()
            summ_rows.append({"arm": arm, "target": tgt, "metric": metric, "mean": v.mean(),
                              "sd": v.std(ddof=1) if len(v) > 1 else np.nan,
                              "pct2_5": v.quantile(0.025), "pct97_5": v.quantile(0.975),
                              "min": v.min(), "max": v.max(), "n_repeats": len(v)})
    pd.DataFrame(summ_rows).to_csv(TABLES_DIR / f"step19_{a.tag}_summary.csv", index=False, encoding="utf-8-sig")

    # A1 contrast: random minus grouped, paired by seed.
    con_rows = []
    for unit in ["L", "C"]:
        rnd, grp = f"{unit}-rand", f"{unit}-group"
        if rnd not in arms or grp not in arms:
            continue
        for tgt in per["target"].unique():
            m = per[(per["arm"] == rnd) & (per["target"] == tgt)].set_index("seed")["mean_fold_auroc"]
            n = per[(per["arm"] == grp) & (per["target"] == tgt)].set_index("seed")["mean_fold_auroc"]
            diff = (m - n).dropna()
            con_rows.append({"unit": "lesion" if unit == "L" else "case", "target": tgt,
                             "random_mean": m.mean(), "grouped_mean": n.mean(),
                             "mean_difference": diff.mean(), "min_difference": diff.min(),
                             "max_difference": diff.max(), "n_random_higher": int((diff > 0).sum()),
                             "n_pairs": len(diff),
                             "inflation_demonstrated": bool(diff.mean() >= 0.05 and (diff > 0).sum() >= 0.9 * len(diff))})
    pd.DataFrame(con_rows).to_csv(TABLES_DIR / f"step19_{a.tag}_contrast.csv", index=False, encoding="utf-8-sig")

    # A2 cross-fitted patient scores (grouped arms): average each patient's score over repeats,
    # then a patient bootstrap.
    rng = np.random.default_rng(cfg.global_seed)
    boot_rows = []
    for (arm, target), frames in pat_store.items():
        if not arm.endswith("group"):
            continue
        for rule in ["mean", "max", "largest"]:
            s = pd.concat([f[rule] for f in frames], axis=1).mean(axis=1)
            yv = frames[0]["y"].reindex(s.index).values
            pv = s.values
            point = safe_auc(yv, pv)
            bs = []
            for _ in range(2000):
                idx = rng.integers(0, len(yv), len(yv))
                if len(np.unique(yv[idx])) > 1:
                    bs.append(roc_auc_score(yv[idx], pv[idx]))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            boot_rows.append({"arm": arm, "target": DISPLAY[target], "rule": rule,
                              "primary_rule": rule == "mean", "n_patients": len(yv),
                              "cross_fitted_auroc": point, "boot_ci_lo": float(lo), "boot_ci_hi": float(hi),
                              "ci_excludes_0_5": bool(lo > 0.5 or hi < 0.5), "n_bootstrap": len(bs)})
    pd.DataFrame(boot_rows).to_csv(TABLES_DIR / f"step19_{a.tag}_patient_crossfitted.csv", index=False,
                                   encoding="utf-8-sig")

    # Secondary Taha-like Random Forest arm on lesion rows.
    rf_rows = []
    if a.rf_seeds > 0:
        t1 = time.time()
        rf_tasks, rf_meta = [], []
        for target in TARGETS:
            d, X, y, groups, _, _ = prepare(units["L"], target)
            for grouped in (False, True):
                for s in range(a.rf_seeds):
                    rf_tasks.append(delayed(rf_once)(X, y, groups, grouped, cfg.global_seed + s))
                    rf_meta.append((DISPLAY[target], "grouped" if grouped else "random", cfg.global_seed + s))
        rf_res = Parallel(n_jobs=a.jobs, backend="loky")(rf_tasks)
        rf_rows = [{"target": t, "splitter": sp, "seed": s, "test_auroc": v}
                   for (t, sp, s), v in zip(rf_meta, rf_res)]
        pd.DataFrame(rf_rows).to_csv(TABLES_DIR / f"step19_{a.tag}_rf_single_split.csv", index=False,
                                     encoding="utf-8-sig")
        log.info("Taha-like RF arm: %d fits in %.1f min", len(rf_tasks), (time.time() - t1) / 60)

    META_DIR.mkdir(parents=True, exist_ok=True)
    json.dump({"tag": a.tag, "arms": arms, "repeats": a.repeats, "rf_seeds": a.rf_seeds,
               "outer_splits": cfg.outer_cv_splits, "inner_splits": cfg.inner_cv_splits,
               "seed_base": cfg.global_seed, "batch_column": BATCH_COL,
               "lesion_features": len(units["L"]["feats"]), "case_features": len(units["C"]["feats"]),
               "case_feature_columns": units["C"]["feats"],
               "protocol": "docs/PROTOCOL_added_analyses.md",
               "runtime_minutes": round((time.time() - t0) / 60, 1)},
              open(META_DIR / f"step19_{a.tag}.json", "w", encoding="utf-8"), indent=1)

    for _, r in pd.DataFrame(con_rows).iterrows():
        log.info("A1 %-6s %-4s random %.3f | grouped %.3f | diff %+.3f (%d/%d random higher)",
                 r["unit"], r["target"], r["random_mean"], r["grouped_mean"], r["mean_difference"],
                 r["n_random_higher"], r["n_pairs"])
    for r in boot_rows:
        if r["primary_rule"]:
            log.info("A2 %-7s %-4s cross-fitted patient AUROC %.3f (95%% CI %.3f-%.3f)",
                     r["arm"], r["target"], r["cross_fitted_auroc"], r["boot_ci_lo"], r["boot_ci_hi"])
    log.info("done in %.1f min -> %s", (time.time() - t0) / 60, TABLES_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
