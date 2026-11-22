"""
step_23_partitions.py — fold assignments for every reported cross-validation run (protocol analysis A6).

WHY THIS FILE EXISTS
--------------------
The evaluation should be reproducible exactly. Seeds are published, but a reader
should not have to re-run the pipeline to see which patient sat in which fold. This script
re-derives the splitters from their seeds, without fitting any model, and writes:

  results/partitions/step05b_primary_partitions.csv
      target, repeat, seed, patient, outer_fold, inner_fold (inner fold of the patient inside its
      outer training set; -1 for the outer fold's held-out patients). Splitters exactly as in
      step 05b: StratifiedKFold(5, shuffle, seed) outer and StratifiedKFold(n_inner, shuffle,
      seed + fold) inner, on the train+valid patients with a label for that receptor.
  results/partitions/step19_unitdemo_partitions.csv
      arm, target, repeat, seed, row_id, patient, outer_fold for analyses A1/A2, copied from the
      fold column the step-19 run itself recorded (no re-derivation needed).
  results/partitions/README.md

A self-check confirms that the per-receptor patient counts equal those of the cross-fitted
analysis (step05b_patient_bootstrap_primary.csv).

Protocol: docs/PROTOCOL_added_analyses.md, A6.

Usage:
    python src/analysis/step_23_partitions.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.modeling.step_05_tabular_modeling import Step05Config  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = _ROOT / "bcbm_project"
INPUT = PROJECT_ROOT / "data" / "processed" / "analysis_ready_step04_patient_level.csv"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
OUT_DIR = PROJECT_ROOT / "results" / "partitions"
TARGETS = {"ER_bin": "ER", "PR_bin": "PR", "HER2_bin": "HER2"}
REPEATS = 10


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("step_23")


def step05b_partitions(cfg: Step05Config) -> pd.DataFrame:
    df = pd.read_csv(INPUT, low_memory=False)
    rows = []
    for target, disp in TARGETS.items():
        d = df[~df[target].isna()].copy()
        d[target] = d[target].astype(int)
        mask = d["split"].isin(["train", "valid"])
        y = d.loc[mask, target].reset_index(drop=True)
        pid = d.loc[mask, "patient_base"].reset_index(drop=True).astype(str)
        n_outer = max(min(cfg.outer_cv_splits, int(y.value_counts().min())), 2)
        for r in range(REPEATS):
            seed = cfg.global_seed + 1000 * r
            outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
            for fold_i, (tr, va) in enumerate(outer.split(y.to_frame(), y)):
                y_otr = y.iloc[tr]
                n_inner = max(min(cfg.inner_cv_splits, int(y_otr.value_counts().min()), len(y_otr)), 2)
                inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed + fold_i)
                inner_of = {}
                for j, (_, iva) in enumerate(inner.split(y_otr.to_frame(), y_otr)):
                    for k in iva:
                        inner_of[tr[k]] = j
                for i in va:
                    rows.append({"target": disp, "repeat": r, "seed": seed, "patient": pid[i],
                                 "y_true": int(y[i]), "outer_fold": fold_i, "role": "outer_test",
                                 "inner_fold": -1})
                # inner assignment is recorded per outer fold for the training patients
                for i, j in inner_of.items():
                    rows.append({"target": disp, "repeat": r, "seed": seed, "patient": pid[i],
                                 "y_true": int(y[i]), "outer_fold": fold_i, "role": "outer_train",
                                 "inner_fold": j})
    return pd.DataFrame(rows)


def main() -> int:
    log = setup_logger()
    cfg = Step05Config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    p = step05b_partitions(cfg)
    boot = pd.read_csv(TABLES_DIR / "step05b_patient_bootstrap_primary.csv")
    boot = boot[boot["feature_set_variant"] == "radiomics_pure"].set_index("target_display")
    test_rows = p[p["role"] == "outer_test"]
    for disp in TARGETS.values():
        g = test_rows[test_rows["target"] == disp]
        per_rep = g.groupby("repeat")["patient"].nunique()
        expected = int(boot.loc[disp, "n_patients"])
        if not (per_rep == expected).all() or g.groupby(["repeat", "patient"]).size().max() != 1:
            raise SystemExit(f"{disp}: partition check failed (expected {expected} patients, each tested once)")
        log.info("%-4s step05b partitions: %d patients x %d repeats, each tested exactly once", disp,
                 expected, per_rep.size)
    p.to_csv(OUT_DIR / "step05b_primary_partitions.csv", index=False, encoding="utf-8-sig")

    s19 = pd.read_csv(TABLES_DIR / "step19_unitdemo_oof_and_partitions.csv",
                      usecols=["arm", "target", "repeat", "seed", "row_id", "patient", "fold", "y_true"])
    s19 = s19.rename(columns={"fold": "outer_fold"})
    s19.to_csv(OUT_DIR / "step19_unitdemo_partitions.csv", index=False, encoding="utf-8-sig")
    log.info("step19 partitions: %d rows (%s)", len(s19), sorted(s19["arm"].unique()))

    (OUT_DIR / "README.md").write_text(
        "# Fold assignments\n\n"
        "`step05b_primary_partitions.csv`: primary repeated nested CV (step 05b, tag `primary`). One row "
        "per patient, receptor, repeat and outer fold. `role = outer_test` rows give the outer fold in "
        "which the patient was held out; `role = outer_train` rows give the patient's inner fold "
        "within that outer fold's training set. Seed of repeat r is 42 + 1000·r; the inner splitter "
        "of outer fold f uses seed + f. Re-derived from the seeds by `src/analysis/step_23_partitions.py`; "
        "no model was refitted.\n\n"
        "`step19_unitdemo_partitions.csv`: analyses A1/A2 (lesion and case rows, random and "
        "patient-grouped arms), outer fold of every row as recorded by `src/analysis/step_19_unit_of_analysis.py`.\n",
        encoding="utf-8")
    log.info("wrote %s", OUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
