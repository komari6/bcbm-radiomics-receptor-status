"""
step_24_r1_supplementary.py — four analyses added after the 13-14 November 2026 rerun. They
were not in the protocol and are reported whatever their direction.

WHY THIS FILE EXISTS
--------------------
Three questions remained open:

(a) Where is "chance" for the primary procedure? Cross-validated AUROC under no signal tends to
    sit below 0.50, so the primary estimates (ER and PR below 0.50) need a null of their own.
      reduced   The reduced pipeline of step 15 run on the ten outer partitions of step 05b
                (same seeds, same splitter), with a permutation null of its ten-partition mean.
      fullnull  A permutation null of the full step 05b procedure (inner model selection over the
                nine fixed configurations, stability selection, in-fold ComBat), radiomics only,
                ten partitions per permutation. Checkpointed per permutation so it can run in the
                background and be stopped or extended without losing work.
(b) capped    The lesion-level arms of step 19 (random vs patient-grouped splitting) with at most
              CAP lesions per patient, so that the seven patients with many follow-up lesions do
              not dominate; pooled out-of-fold lesion AUROC is reported beside the fold mean.
(c) flips     The label-noise model of step 22 with one-sided (directional) discordance, e.g.
              loss of PR expression in the metastasis, beside the symmetric model.

Every label-using part is restricted to the 123 train + valid patients; the locked test stays out.
Fold-internal processing is imported unchanged from steps 05, 05b, 15 and 19.

Usage:
    python src/analysis/step_24_r1_supplementary.py --part flips
    python src/analysis/step_24_r1_supplementary.py --part reduced --n-perm 1000 --jobs 20
    python src/analysis/step_24_r1_supplementary.py --part capped --jobs 20
    python src/analysis/step_24_r1_supplementary.py --part fullnull --n-perm 50 --jobs 22
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.modeling.step_05_tabular_modeling import (  # noqa: E402
    Step05Config,
    get_batch_labels,
    harmonize_pair_if_needed,
)
from src.modeling import step_05b_repeated_nested_cv as s05b  # noqa: E402
from src.analysis import step_15_robustness_null as s15  # noqa: E402
from src.analysis import step_19_unit_of_analysis as s19  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = _ROOT / "bcbm_project"
INPUT = PROJECT_ROOT / "data" / "processed" / "analysis_ready_step04_patient_level.csv"
FEATURE_DICT = PROJECT_ROOT / "metadata" / "step04_feature_dictionary.json"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
META_DIR = PROJECT_ROOT / "metadata"

TARGETS = ["ER_bin", "PR_bin", "HER2_bin"]
DISPLAY = {"ER_bin": "ER", "PR_bin": "PR", "HER2_bin": "HER2"}
N_REPEATS = 10            # the ten outer partitions of step 05b
CAP = 20                  # lesions per patient in the capped arms
FLIP_EPS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
FLIP_TRUE_AUCS = [0.65, 0.75, 0.85]
FLIP_N = 200_000


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("step_24")


# ----------------------------------------------------------------------------- shared data

def load_primary(cfg) -> Dict[str, Dict]:
    """Radiomics-only train+valid data per target, prepared exactly as step 05b prepares it."""
    df = pd.read_csv(INPUT, low_memory=False)
    fd = json.load(open(FEATURE_DICT, encoding="utf-8"))
    cols = [c for c in fd["radiomic_block_columns"] if c in df.columns]
    batch_all = get_batch_labels(df)
    out = {}
    for target in TARGETS:
        d = df[~df[target].isna()].copy()
        d[target] = d[target].astype(int)
        for c in cols:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d[cols] = d[cols].replace([np.inf, -np.inf], np.nan)
        mask = d["split"].isin(["train", "valid"])
        X = d.loc[mask, cols].reset_index(drop=True)
        y = d.loc[mask, target].reset_index(drop=True)
        use_combat = batch_all is not None
        if use_combat:
            bl = pd.Series(batch_all, index=df.index).reindex(d.index).fillna(-1).astype(int)
            batch = bl.loc[mask].values
        else:
            batch = np.zeros(len(y), dtype=int)
        n_outer = max(min(cfg.outer_cv_splits, int(y.value_counts().min())), 2)
        out[target] = {"X": X, "y": y, "batch": batch, "cols": cols, "use_combat": use_combat,
                       "n_outer": n_outer}
    return out


def draw_perms(y: np.ndarray, n: int, target_idx: int) -> List[np.ndarray]:
    """Permutations drawn serially from a per-target generator: any smaller count reproduces the
    leading draws of a larger one, independent of scheduling."""
    rng = np.random.default_rng([getattr(Step05Config(), "global_seed", 42), target_idx])
    return [rng.permutation(y) for _ in range(n)]


def perm_p(null: np.ndarray, obs: float) -> float:
    return float((np.sum(null >= obs) + 1) / (len(null) + 1))


# ----------------------------------------------------------------------------- (a) reduced pipeline

def reduced_partitions(X, y_vals, batch, cols, use_combat, n_outer, seed_base) -> Dict:
    """Step 15's reduced pipeline on the ten step-05b partitions; returns the ten-partition mean
    of the mean outer-fold AUROC (the step 05b endpoint) and of the pooled out-of-fold AUROC."""
    s5 = s15.load_step05()
    cfg = s5.Step05Config()
    quiet = logging.getLogger("quiet")
    y = pd.Series(y_vals)
    fold_means, pooled, oofs = [], [], []
    for r in range(N_REPEATS):
        seed = seed_base + 1000 * r
        skf = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
        oof = np.full(len(y), np.nan)
        aucs = []
        for tr, va in skf.split(X, y):
            Xtr, Xva = X.iloc[tr], X.iloc[va]
            if use_combat:
                Xtr, Xva = harmonize_pair_if_needed(Xtr, Xva, batch[tr], batch[va], cols, True, quiet)
            try:
                pipe = s15.light_pipeline(seed, cfg)
                pipe.fit(Xtr, y.iloc[tr])
                p = pipe.predict_proba(Xva)[:, 1]
                oof[va] = p
                aucs.append(float(roc_auc_score(y.iloc[va], p)) if y.iloc[va].nunique() > 1 else np.nan)
            except Exception:
                aucs.append(np.nan)
        fold_means.append(float(np.nanmean(aucs)))
        ok = ~np.isnan(oof)
        pooled.append(float(roc_auc_score(y[ok], oof[ok])) if ok.sum() and y[ok].nunique() > 1 else np.nan)
        oofs.append(oof)
    return {"mean_fold": float(np.mean(fold_means)), "pooled": float(np.nanmean(pooled)),
            "per_partition": fold_means, "oof_mean": np.nanmean(np.vstack(oofs), axis=0)}


def run_reduced(a, log) -> None:
    cfg = Step05Config()
    data = load_primary(cfg)
    seed_base = getattr(cfg, "global_seed", 42)
    rows, per_rows = [], []
    for ti, target in enumerate(TARGETS):
        t0 = time.time()
        d = data[target]
        obs = reduced_partitions(d["X"], d["y"].values, d["batch"], d["cols"], d["use_combat"],
                                 d["n_outer"], seed_base)
        cf = float(roc_auc_score(d["y"].values, obs["oof_mean"]))
        perms = draw_perms(d["y"].values, a.n_perm, ti)
        res = Parallel(n_jobs=a.jobs, backend="loky", inner_max_num_threads=1)(
            delayed(reduced_partitions)(d["X"], yp, d["batch"], d["cols"], d["use_combat"], d["n_outer"],
                                        seed_base) for yp in perms)
        null_fold = np.array([r["mean_fold"] for r in res])
        null_pool = np.array([r["pooled"] for r in res])
        for i, r in enumerate(res):
            per_rows.append({"target": DISPLAY[target], "permutation": i, "mean_fold_auroc": r["mean_fold"],
                             "pooled_oof_auroc": r["pooled"]})
        rows.append({"target": DISPLAY[target], "n_patients": len(d["y"]), "n_partitions": N_REPEATS,
                     "observed_mean_fold_auroc": obs["mean_fold"],
                     "observed_partition_min": min(obs["per_partition"]),
                     "observed_partition_max": max(obs["per_partition"]),
                     "observed_pooled_oof_auroc": obs["pooled"], "observed_cross_fitted_auroc": cf,
                     "null_mean": float(null_fold.mean()), "null_sd": float(null_fold.std(ddof=1)),
                     "null_p2_5": float(np.percentile(null_fold, 2.5)),
                     "null_p95": float(np.percentile(null_fold, 95)),
                     "null_p97_5": float(np.percentile(null_fold, 97.5)),
                     "p_value": perm_p(null_fold, obs["mean_fold"]),
                     "null_pooled_mean": float(np.nanmean(null_pool)),
                     "p_value_pooled": perm_p(null_pool[~np.isnan(null_pool)], obs["pooled"]),
                     "n_permutations": len(null_fold)})
        log.info("reduced %-4s observed %.3f (partitions %.3f-%.3f) | null %.3f (p95 %.3f) | p=%.3f | %.1f min",
                 DISPLAY[target], obs["mean_fold"], rows[-1]["observed_partition_min"],
                 rows[-1]["observed_partition_max"], rows[-1]["null_mean"], rows[-1]["null_p95"],
                 rows[-1]["p_value"], (time.time() - t0) / 60)
    pd.DataFrame(rows).to_csv(TABLES_DIR / "step24_reduced_partitions_null.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(per_rows).to_csv(TABLES_DIR / "step24_reduced_partitions_null_draws.csv", index=False,
                                  encoding="utf-8-sig")


# ----------------------------------------------------------------------------- (a) full-procedure null

def run_fullnull(a, log) -> None:
    """Permutation null of the full step 05b procedure, radiomics only. One row per (target,
    permutation) is appended to the draws file as soon as its ten partitions finish."""
    cfg = Step05Config()
    data = load_primary(cfg)
    seed_base = getattr(cfg, "global_seed", 42)
    draws_path = TABLES_DIR / "step24_full_procedure_null_draws.csv"
    done = set()
    if draws_path.exists():
        prev = pd.read_csv(draws_path)
        done = set(zip(prev["target"], prev["permutation"]))
    perms = {t: draw_perms(data[t]["y"].values, a.n_perm, ti) for ti, t in enumerate(TARGETS)}
    todo = [(t, i) for i in range(a.n_perm) for t in TARGETS if (DISPLAY[t], i) not in done]
    log.info("full-procedure null: %d of %d (target, permutation) cells to run, %d tasks each",
             len(todo), a.n_perm * len(TARGETS), N_REPEATS)
    # One flat task list over every (cell, partition), dispatched in cell order and collected as each
    # task finishes, so no worker waits at a batch boundary (2026-11-15; the earlier 3-cell batches
    # left 14 of 22 workers idle in each second wave). Each finished partition is appended to the
    # partials file at once (2026-11-17), so stopping the run loses only the partitions still running;
    # a cell goes to the draws file once all ten of its partitions are in.
    partials_path = TABLES_DIR / "step24_full_procedure_null_partials.csv"
    pending: Dict[Tuple[str, int], Dict[int, Dict]] = {}
    todo_keys = set(todo)
    if partials_path.exists():
        pp = pd.read_csv(partials_path, on_bad_lines="skip").dropna()  # a stop can cut the last line
        for row in pp.itertuples(index=False):
            key = (row.target, int(row.permutation))
            if key in todo_keys:
                pending.setdefault(key, {})[int(row.partition)] = {
                    "mean_fold_auroc": row.mean_fold_auroc, "pooled_oof_auroc": row.pooled_oof_auroc}
    n_written = 0

    def write_cell(t, i):
        nonlocal n_written
        res = list(pending.pop((t, i)).values())
        row = {"target": DISPLAY[t], "permutation": i,
               "mean_fold_auroc": float(np.mean([x["mean_fold_auroc"] for x in res])),
               "pooled_oof_auroc": float(np.nanmean([x["pooled_oof_auroc"] for x in res]))}
        pd.DataFrame([row]).to_csv(draws_path, mode="a", header=not draws_path.exists(), index=False,
                                   encoding="utf-8")
        n_written += 1
        log.info("full null: cell (%s, %d) written | %d/%d cells | %.1f min elapsed", DISPLAY[t], i,
                 n_written, len(todo), (time.time() - t0) / 60)

    t0 = time.time()
    n_saved = sum(len(v) for v in pending.values())
    # Cells whose ten partitions were all saved before a stop are written straight away.
    written_now = [k for k, v in pending.items() if len(v) >= N_REPEATS]
    for key in written_now:
        write_cell(*key)
    tasks = []
    for (t, i) in todo:
        if (t, i) in written_now:
            continue
        d = data[t]
        yp = pd.Series(perms[t][i])
        for r in range(N_REPEATS):
            if r in pending.get((t, i), {}):
                continue
            tasks.append(delayed(_fullnull_task)(t, i, r, d["X"], yp, d["batch"], d["cols"], d["use_combat"],
                                                 cfg, d["n_outer"], seed_base + 1000 * r))
    log.info("full null: %d partitions restored from %s, %d partitions to run", n_saved, partials_path.name,
             len(tasks))
    for t, i, r, res in Parallel(n_jobs=a.jobs, backend="loky", inner_max_num_threads=1,
                                 return_as="generator_unordered")(tasks):
        part = {"target": t, "permutation": i, "partition": r,
                "mean_fold_auroc": float(res["mean_fold_auroc"]), "pooled_oof_auroc": float(res["pooled_oof_auroc"])}
        pd.DataFrame([part]).to_csv(partials_path, mode="a", header=not partials_path.exists(), index=False,
                                    encoding="utf-8")
        pending.setdefault((t, i), {})[r] = part
        if len(pending[(t, i)]) >= N_REPEATS:
            write_cell(t, i)
    summarize_fullnull(log)


def _fullnull_task(t, i, r, X, yp, batch, cols, use_combat, cfg, n_outer, seed):
    """One partition of one permuted cell; returns its key so results can be collected out of order."""
    return t, i, r, s05b.one_repeat(X, yp, batch, cols, use_combat, cfg, n_outer, seed)


def summarize_fullnull(log) -> None:
    draws = pd.read_csv(TABLES_DIR / "step24_full_procedure_null_draws.csv")
    obs = pd.read_csv(TABLES_DIR / "step05b_repeated_nested_cv_primary_summary.csv")
    obs = obs[obs["feature_set_variant"] == "radiomics_pure"].set_index("target_display")["mean"]
    rows = []
    for t, g in draws.groupby("target"):
        v = g["mean_fold_auroc"].values
        rows.append({"target": t, "observed_mean_fold_auroc": float(obs[t]), "null_mean": float(v.mean()),
                     "null_sd": float(v.std(ddof=1)) if len(v) > 1 else np.nan,
                     "null_p2_5": float(np.percentile(v, 2.5)), "null_p95": float(np.percentile(v, 95)),
                     "null_p97_5": float(np.percentile(v, 97.5)), "p_value": perm_p(v, float(obs[t])),
                     "observed_minus_null_mean": float(obs[t] - v.mean()), "n_permutations": len(v)})
        log.info("full null %-4s observed %.3f | null %.3f (SD %.3f, p95 %.3f) | p=%.3f | n=%d",
                 t, obs[t], v.mean(), rows[-1]["null_sd"], rows[-1]["null_p95"], rows[-1]["p_value"], len(v))
    out = pd.DataFrame(rows)
    # Holm correction across the three receptors (PROTOCOL_R1.md section S24, 2026-11-18).
    order = out["p_value"].sort_values().index
    m, run = len(out), 0.0
    adj = pd.Series(np.nan, index=out.index)
    for k, idx in enumerate(order):
        run = max(run, min(1.0, (m - k) * out.loc[idx, "p_value"]))
        adj[idx] = run
    out["p_value_holm"] = adj
    for r in out.itertuples(index=False):
        log.info("full null %-4s p=%.3f | Holm-adjusted p=%.3f", r.target, r.p_value, r.p_value_holm)
    out.to_csv(TABLES_DIR / "step24_full_procedure_null.csv", index=False, encoding="utf-8-sig")


# ----------------------------------------------------------------------------- (b) capped lesion arms

def cap_lesions(les: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """Keep at most `cap` lesion rows per patient, chosen at random without looking at labels or
    features; patients are visited in sorted order so the draw is reproducible."""
    rng = np.random.default_rng(seed)
    keep = []
    for pid in sorted(les["patient_base"].unique()):
        idx = np.flatnonzero(les["patient_base"].values == pid)
        keep.extend(idx if len(idx) <= cap else np.sort(rng.choice(idx, cap, replace=False)))
    return les.iloc[np.sort(np.array(keep))].reset_index(drop=True)


def run_capped(a, log) -> None:
    cfg = Step05Config()
    units = s19.load_units(log)
    les = units["L"]["df"]
    capped = cap_lesions(les, CAP, getattr(cfg, "global_seed", 42))
    log.info("capped lesion table: %d -> %d rows, %d patients (cap %d)", len(les), len(capped),
             capped["patient_base"].nunique(), CAP)
    unit = {"df": capped, "feats": units["L"]["feats"]}
    tasks, meta, prepared = [], [], {}
    for arm in ["L-rand", "L-group"]:
        grouped = arm.endswith("group")
        for target in TARGETS:
            d, X, y, groups, batch, cols = s19.prepare(unit, target)
            prepared[(arm, target)] = d
            for r in range(N_REPEATS):
                seed = getattr(cfg, "global_seed", 42) + 1000 * r
                tasks.append(delayed(s19.one_repeat)(X, y, groups, batch, cols, grouped, cfg, seed))
                meta.append((arm, target, r, seed, len(y), d["patient_base"].nunique()))
    t0 = time.time()
    res = Parallel(n_jobs=a.jobs, backend="loky", inner_max_num_threads=1, verbose=5)(tasks)
    log.info("capped arms finished in %.1f min", (time.time() - t0) / 60)
    per = []
    for (arm, target, r, seed, n_rows, n_pat), out in zip(meta, res):
        d = prepared[(arm, target)]
        ps = s19.patient_scores(d, target, out["oof"])
        per.append({"arm": arm + f"-cap{CAP}", "target": DISPLAY[target], "repeat": r, "seed": seed,
                    "n_rows": n_rows, "n_patients": n_pat, "mean_fold_auroc": out["mean_fold_auroc"],
                    "pooled_oof_auroc_rows": s19.safe_auc(d[target].astype(int), out["oof"]),
                    "patient_auroc_mean": s19.safe_auc(ps["y"], ps["mean"])})
    per = pd.DataFrame(per)
    per.to_csv(TABLES_DIR / f"step24_capped{CAP}_per_repeat.csv", index=False, encoding="utf-8-sig")
    summ = per.groupby(["arm", "target"]).agg(
        n_rows=("n_rows", "first"), n_patients=("n_patients", "first"),
        mean_fold_auroc=("mean_fold_auroc", "mean"), mean_fold_min=("mean_fold_auroc", "min"),
        mean_fold_max=("mean_fold_auroc", "max"), pooled_oof_auroc=("pooled_oof_auroc_rows", "mean"),
        patient_auroc_mean=("patient_auroc_mean", "mean")).reset_index()
    con = []
    for tgt in per["target"].unique():
        m = per[(per["arm"] == f"L-rand-cap{CAP}") & (per["target"] == tgt)].set_index("seed")
        n = per[(per["arm"] == f"L-group-cap{CAP}") & (per["target"] == tgt)].set_index("seed")
        for metric in ["mean_fold_auroc", "pooled_oof_auroc_rows"]:
            diff = (m[metric] - n[metric]).dropna()
            con.append({"target": tgt, "metric": metric, "random_mean": m[metric].mean(),
                        "grouped_mean": n[metric].mean(), "mean_difference": diff.mean(),
                        "min_difference": diff.min(), "max_difference": diff.max(),
                        "n_random_higher": int((diff > 0).sum()), "n_pairs": len(diff)})
            log.info("capped %-4s %-22s random %.3f | grouped %.3f | diff %+.3f (%d/%d)", tgt, metric,
                     con[-1]["random_mean"], con[-1]["grouped_mean"], con[-1]["mean_difference"],
                     con[-1]["n_random_higher"], con[-1]["n_pairs"])
    summ.to_csv(TABLES_DIR / f"step24_capped{CAP}_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(con).to_csv(TABLES_DIR / f"step24_capped{CAP}_contrast.csv", index=False, encoding="utf-8-sig")


# ----------------------------------------------------------------------------- (c) one-sided flips

def run_flips(a, log) -> None:
    """Symmetric vs one-sided label flips under the binormal model of step 22.

    'neg_to_pos': a fraction eps of truly negative lesions carry a positive label (for example the
    label comes from a PR-positive primary and the metastasis has lost PR). 'pos_to_neg': a
    fraction eps of truly positive lesions carry a negative label (for example HER2 gained in the
    metastasis). eps is the class-conditional flip rate. Closed forms: with one-sided flips the
    contaminated class is a mixture, so observed AUROC = 0.5 + (A - 0.5) * w, where w is the share
    of true members in the contaminated observed class."""
    rng = np.random.default_rng(getattr(Step05Config(), "global_seed", 42))
    tv = pd.read_csv(INPUT, usecols=["split"] + TARGETS)
    tv = tv[tv["split"].isin(["train", "valid"])]
    prevs = {"0.50": 0.5}
    prevs.update({f"{DISPLAY[t]} ({tv[t].mean():.2f})": float(tv[t].mean()) for t in TARGETS})
    rows = []
    for prev_label, pi in prevs.items():
        for A in FLIP_TRUE_AUCS:
            dshift = np.sqrt(2) * norm.ppf(A)
            y = (rng.random(FLIP_N) < pi).astype(int)
            s = rng.normal(0.0, 1.0, FLIP_N) + dshift * y
            u = rng.random(FLIP_N)
            for eps in FLIP_EPS:
                for mode in ["symmetric", "neg_to_pos", "pos_to_neg"]:
                    if mode == "symmetric":
                        flip = u < eps
                        w = (1 - 2 * eps) if pi == 0.5 else np.nan
                    elif mode == "neg_to_pos":
                        flip = (u < eps) & (y == 0)
                        w = pi / (pi + eps * (1 - pi))
                    else:
                        flip = (u < eps) & (y == 1)
                        w = (1 - pi) / ((1 - pi) + eps * pi)
                    y_obs = np.where(flip, 1 - y, y)
                    obs = float(roc_auc_score(y_obs, s))
                    rows.append({"prevalence": prev_label, "pi": pi, "true_auroc": A, "epsilon": eps,
                                 "mode": mode, "observed_auroc": obs,
                                 "closed_form_auroc": 0.5 + (A - 0.5) * w if not np.isnan(w) else np.nan,
                                 "overall_flip_fraction": float(flip.mean())})
    out = pd.DataFrame(rows)
    out["abs_diff_sim_vs_closed_form"] = (out["observed_auroc"] - out["closed_form_auroc"]).abs()
    out.to_csv(TABLES_DIR / "step24_label_flips_one_sided.csv", index=False, encoding="utf-8-sig")
    log.info("flips: %d rows; max |sim - closed form| = %.4f", len(out), out["abs_diff_sim_vs_closed_form"].max())
    show = out[(out["true_auroc"] == 0.75) & (out["epsilon"].isin([0.1, 0.2, 0.3]))]
    log.info("\n%s", show.pivot_table(index=["prevalence", "epsilon"], columns="mode",
                                      values="observed_auroc").round(3).to_string())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["reduced", "fullnull", "fullnull-summary", "capped", "flips"])
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=20)
    a = ap.parse_args()
    log = setup_logger()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    {"reduced": run_reduced, "fullnull": run_fullnull, "capped": run_capped, "flips": run_flips,
     "fullnull-summary": lambda a_, l_: summarize_fullnull(l_)}[a.part](a, log)
    json.dump({"part": a.part, "n_perm": a.n_perm, "jobs": a.jobs, "cap": CAP, "n_repeats": N_REPEATS,
               "runtime_minutes": round((time.time() - t0) / 60, 1),
               "source": "added after the 2026-11-13/14 rerun; fullnull reporting rule in docs/PROTOCOL_added_analyses.md, S24"},
              open(META_DIR / f"step24_{a.part}.json", "w", encoding="utf-8"), indent=1)
    log.info("step 24 [%s] done in %.1f min", a.part, (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
