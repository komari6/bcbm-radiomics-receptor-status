"""
step_21_detectable_effect.py — detectable effect at the patient level (protocol analysis A4).

WHY THIS FILE EXISTS
--------------------
Nominal power of 0.95-0.98 for AUROC 0.70 in the out-of-fold pool assumes independent
predictions, which the five fold models do not provide. This script puts three
quantities side by side for each receptor, with patients as the unit:

  1. Hanley-McNeil standard error at AUROC 0.50 and the minimum AUROC detectable with 80% power
     (alpha 0.05), from the per-receptor patient class counts of the cross-fitted analysis;
  2. the empirical standard error implied by the existing 2,000-sample patient bootstrap of the
     cross-fitted AUROC (CI width / 3.92), which carries the same patient-level sampling variance
     but not the variance between fold models;
  3. the upper bound of that bootstrap CI: the largest AUROC the data remain compatible with.

The step-14 power file is read, never rewritten. Hanley-McNeil functions are imported from
step 14 so that the arithmetic is the same.

Protocol: docs/PROTOCOL_added_analyses.md, A4.

Usage:
    python src/analysis/step_21_detectable_effect.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis.step_14_learning_curve_power import (  # noqa: E402
    hanley_mcneil_se,
    min_detectable_auc,
    power_auc,
)

_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = _ROOT / "bcbm_project"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
META_DIR = PROJECT_ROOT / "metadata"
BOOT = TABLES_DIR / "step05b_patient_bootstrap_primary.csv"
SCORES = TABLES_DIR / "step05b_crossfitted_scores_primary.csv"  # one primary run since the 2026-11-13 rebuild
PRIMARY_VARIANT = "radiomics_pure"


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("step_21")


def class_counts(target: str, n_patients: int) -> tuple[int, int]:
    """Patient class counts for a receptor, read from a step-05b cross-fitted score file.

    The primary run did not save per-patient scores, but every feature-set variant of a receptor
    uses the same labelled train+valid patients, so any variant of the primary score file gives the counts.
    The patient total is checked against the primary bootstrap file.
    """
    s = pd.read_csv(SCORES)
    s = s[s["target"] == target]
    s = s[s["feature_set_variant"] == s["feature_set_variant"].iloc[0]]
    if len(s) != n_patients:
        raise SystemExit(f"{target}: {len(s)} scored patients vs {n_patients} in the bootstrap file")
    n_pos = int(s["y_true"].sum())
    return n_pos, int(len(s) - n_pos)


def main() -> int:
    log = setup_logger()
    boot = pd.read_csv(BOOT)
    boot = boot[boot["feature_set_variant"] == PRIMARY_VARIANT]
    rows = []
    for _, b in boot.iterrows():
        n_pos, n_neg = class_counts(b["target"], int(b["n_patients"]))
        se_hm0 = hanley_mcneil_se(0.5, n_pos, n_neg)
        se_boot = (b["boot_ci_hi"] - b["boot_ci_lo"]) / 3.92
        rows.append({
            "target": b["target"], "target_display": b["target_display"],
            "feature_set_variant": PRIMARY_VARIANT, "unit": "patient_unit",
            "n_patients": int(b["n_patients"]), "n_pos": n_pos, "n_neg": n_neg,
            "hm_se_at_0_50": se_hm0,
            "hm_min_detectable_auc_80pct": min_detectable_auc(n_pos, n_neg),
            "hm_power_auc_0_70": power_auc(0.70, n_pos, n_neg),
            "cross_fitted_auroc": b["cross_fitted_auroc"],
            "boot_ci_lo": b["boot_ci_lo"], "boot_ci_hi": b["boot_ci_hi"],
            "boot_implied_se": se_boot, "boot_se_over_hm_se": se_boot / se_hm0,
            "largest_auc_compatible_95ci": b["boot_ci_hi"],
        })
        r = rows[-1]
        log.info("%-4s n=%d (%d+/%d-) | HM SE %.4f, MDA80 %.3f, power@0.70 %.3f | boot SE %.4f "
                 "(ratio %.2f) | CI upper %.3f", r["target_display"], r["n_patients"], n_pos, n_neg,
                 se_hm0, r["hm_min_detectable_auc_80pct"], r["hm_power_auc_0_70"], se_boot,
                 r["boot_se_over_hm_se"], r["boot_ci_hi"])
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "step21_detectable_effect.csv", index=False, encoding="utf-8-sig")
    json.dump({"inputs": [BOOT.name, SCORES.name], "feature_set_variant": PRIMARY_VARIANT,
               "method": "Hanley-McNeil (step 14 functions), patients as unit; bootstrap-implied SE = "
                         "(CI hi - CI lo) / 3.92"},
              open(META_DIR / "step21_detectable_effect.json", "w", encoding="utf-8"), indent=1)
    log.info("wrote step21_detectable_effect.csv (%d rows)", len(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
