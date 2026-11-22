"""
step_20_inclusion_comparison.py — included versus excluded patients (protocol analysis A3).

WHY THIS FILE EXISTS
--------------------
Of the 165 patients in BCBM-RadioGenomics, 140 enter the analysis. The question is
whether the other 25 differ in a way that could bias the estimates. This script records why each
of the 25 was excluded, then compares the groups on the variables the public release provides.

Exclusion reasons are read from the step-03 classification, not re-derived:
  - no receptor label of any kind in the released clinical sheet, or
  - labels present but no structure classified as a lesion.

Lesion counts and volumes use the same definition for every group (final_mask_category ==
"lesion", before label filtering), so patients excluded for missing labels can still be
described by their segmentations.

The comparison is descriptive (protocol docs/PROTOCOL_added_analyses.md, A3):
Mann-Whitney U / Fisher exact tests, standardized mean differences, no multiplicity correction.

Usage:
    python src/analysis/step_20_inclusion_comparison.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = _ROOT / "bcbm_project"
FULL_TABLE = PROJECT_ROOT / "data" / "processed" / "analysis_ready_step03_full.csv"
LESION_TABLE = PROJECT_ROOT / "data" / "processed" / "analysis_ready_step03_lesion_only.csv"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
META_DIR = PROJECT_ROOT / "metadata"
RECEPTORS = ["ER", "PR", "HER2"]


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("step_20")


def clean_label(s: pd.Series) -> pd.Series:
    v = s.astype(str).str.strip()
    return v.where(v.isin(["+", "-"]), np.nan)


def patient_table(full: pd.DataFrame, included: set) -> pd.DataFrame:
    """One row per patient: demographics at the first examination, burden, labels, reason."""
    f = full.copy()
    f["pb"] = f["patient_base"].astype(str)
    for r in RECEPTORS:
        f[r] = clean_label(f[r])
    f["is_lesion"] = f["final_mask_category"].astype(str).eq("lesion")
    f["visit_index"] = pd.to_numeric(f["visit_index"], errors="coerce")

    rows = []
    for pb, g in f.groupby("pb"):
        first = g.sort_values("visit_index").iloc[0]
        cases = g.drop_duplicates("case_id")
        les = g[g["is_lesion"]]
        # Same definition as step 04's dominant_field_strength_t: the mode over the patient's
        # lesion rows (all rows when the patient has no lesion mask).
        fs = pd.to_numeric((les if len(les) else g)["Magnetic_Field_Strength_ID"], errors="coerce").dropna()
        labels = {r: (g[r].dropna().iloc[0] if g[r].notna().any() else np.nan) for r in RECEPTORS}
        any_label = any(isinstance(v, str) for v in labels.values())
        if pb in included:
            reason = "included"
        elif not any_label:
            reason = "no_receptor_label"
        elif les.empty:
            reason = "no_lesion_mask"
        else:
            reason = "other"
        rows.append({
            "patient_base": pb,
            "group": "included" if pb in included else "excluded",
            "reason": reason,
            "age_first_exam": pd.to_numeric(first["Age"], errors="coerce"),
            "year_first_exam": pd.to_numeric(first["Year"], errors="coerce"),
            "field_strength_dominant_t": float(fs.mode().iloc[0]) if len(fs) else np.nan,
            "n_examinations": int(cases["case_id"].nunique()),
            "n_lesion_masks": int(len(les)),
            "total_lesion_volume_cc": float(pd.to_numeric(les["mask_volume_cc"], errors="coerce").sum())
            if len(les) else np.nan,
            "sex": str(first["Sex"]).strip(),
            "manufacturer": str(first["Manufacturer"]).strip(),
            **{f"{r}_status": labels[r] for r in RECEPTORS},
        })
    return pd.DataFrame(rows)


def smd_continuous(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna().astype(float), b.dropna().astype(float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / sd) if sd > 0 else np.nan


def smd_binary(p1: float, p2: float) -> float:
    sd = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    return float((p1 - p2) / sd) if sd > 0 else np.nan


def compare(pt: pd.DataFrame, exc_mask: pd.Series, label: str) -> list:
    inc = pt[pt["group"] == "included"]
    exc = pt[exc_mask]
    out = []
    for col in ["age_first_exam", "year_first_exam", "n_examinations", "n_lesion_masks",
                "total_lesion_volume_cc"]:
        a, b = inc[col].dropna(), exc[col].dropna()
        p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue) if len(a) and len(b) else np.nan
        out.append({"comparison": label, "variable": col, "type": "continuous",
                    "included_n": len(a), "included_median": a.median(),
                    "included_q1": a.quantile(.25), "included_q3": a.quantile(.75),
                    "excluded_n": len(b), "excluded_median": b.median() if len(b) else np.nan,
                    "excluded_q1": b.quantile(.25) if len(b) else np.nan,
                    "excluded_q3": b.quantile(.75) if len(b) else np.nan,
                    "smd": smd_continuous(a, b), "test": "Mann-Whitney U", "p_value": p})
    cats = [("field_strength_3t", lambda d: (d["field_strength_dominant_t"] >= 2.5)
             .where(d["field_strength_dominant_t"].notna()))]
    for r in RECEPTORS:
        cats.append((f"{r}_positive", lambda d, r=r: d[f"{r}_status"].map({"+": True, "-": False})))
    for name, fn in cats:
        a, b = fn(inc).dropna().astype(bool), fn(exc).dropna().astype(bool)
        if len(a) and len(b):
            p = float(fisher_exact([[a.sum(), (~a).sum()], [b.sum(), (~b).sum()]])[1])
            smd = smd_binary(a.mean(), b.mean())
        else:
            p, smd = np.nan, np.nan
        out.append({"comparison": label, "variable": name, "type": "binary",
                    "included_n": len(a), "included_count": int(a.sum()),
                    "included_prop": a.mean() if len(a) else np.nan,
                    "excluded_n": len(b), "excluded_count": int(b.sum()) if len(b) else 0,
                    "excluded_prop": b.mean() if len(b) else np.nan,
                    "smd": smd, "test": "Fisher exact", "p_value": p})
    return out


def main() -> int:
    log = setup_logger()
    full = pd.read_csv(FULL_TABLE, low_memory=False)
    lesion = pd.read_csv(LESION_TABLE, low_memory=False)
    included = set(lesion["patient_base"].astype(str))
    pt = patient_table(full, included)
    counts = pt["reason"].value_counts().to_dict()
    log.info("patients=%d | reasons=%s", len(pt), counts)
    if counts.get("other", 0):
        log.warning("%d excluded patients have no recorded reason", counts["other"])

    rows = compare(pt, pt["group"] == "excluded", "included_vs_all_excluded")
    rows += compare(pt, pt["reason"] == "no_receptor_label", "included_vs_no_receptor_label")
    res = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(TABLES_DIR / "step20_inclusion_comparison.csv", index=False, encoding="utf-8-sig")
    pt[pt["group"] == "excluded"].to_csv(TABLES_DIR / "step20_excluded_patient_reasons.csv",
                                         index=False, encoding="utf-8-sig")
    constants = {c: sorted(pt[c].unique().tolist()) for c in ["sex", "manufacturer"]}
    json.dump({"n_patients": int(len(pt)), "n_included": int((pt["group"] == "included").sum()),
               "exclusion_reasons": {k: int(v) for k, v in counts.items()},
               "excluded_with_lesion_masks": int(((pt["group"] == "excluded") & (pt["n_lesion_masks"] > 0)).sum()),
               "constant_variables": constants},
              open(META_DIR / "step20_inclusion_comparison.json", "w", encoding="utf-8"), indent=1)
    log.info("wrote step20_inclusion_comparison.csv (%d rows) and step20_excluded_patient_reasons.csv", len(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
