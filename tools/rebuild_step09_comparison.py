"""Rewrite step09_final_comparison_all_stages.csv from stored results, without retraining.

The 2026-11-13 rebuild ran step 09 in parallel with step 05, and step 09 built its cross-stage
table before step 05 had written the HER2 radiomic-block result, so the table (and Fig N of S1)
lacked the HER2 tabular point. This script calls step 09's own build_cross_stage_comparison on the
stored validation-selected step-09 rows; nothing is refit.
"""
import logging
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "modeling"))
sys.path.insert(0, os.path.join(ROOT, "src"))
import step_09_hybrid_fusion as s09  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("rebuild_step09_comparison")

T = os.path.join(ROOT, "bcbm_project", "reports", "tables")
res = pd.read_csv(os.path.join(T, "step09_hybrid_model_results.csv"))
sel = res[res["selected_by_validation"].astype(bool)]
rows = [{"target": r["target"], "variant": r["variant"],
         "test_metrics": {"AUROC": r["test_AUROC"], "AUPRC": r["test_AUPRC"], "F1": r["test_F1"],
                          "Balanced_Acc": r["test_Balanced_Acc"]}} for _, r in sel.iterrows()]
comp = s09.build_cross_stage_comparison(rows)
out = os.path.join(T, "step09_final_comparison_all_stages.csv")
comp.to_csv(out, index=False, encoding="utf-8-sig")
logger.info("wrote %s (%d rows)", out, len(comp))
