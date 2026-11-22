#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 10 — Build OpenBTAI External Proxy Dataset
================================================

Purpose
-------
Creates clean analysis-ready OpenBTAI tables with:
  1) clinical primary-cancer labels
  2) breast molecular subtype labels
  3) receptor-proxy labels derived from breast subtype
  4) optional direct receptor labels if RE/RP/HER2 columns contain values
  5) merged radiomics + morphology features
  6) feature dictionary for Step 11

Scientific note
---------------
These are PROXY receptor labels, not direct pathology receptor labels.
Report in the paper as "external proxy radiogenomic validation", not
"direct external ER/PR/HER2 validation".

Inputs (run from the project root)
--------------------------------------------------------------
- OpenBTAI_METS_ClinicalData_Nov2023.xlsx
- OpenBTAI_RADIOMICS.xlsx
- OpenBTAI_MORPHOLOGICAL_MEASUREMENTS.xlsx

Outputs
-------
  bcbm_project/data/external_ready/
    openbtai_breast_only_patient_level_proxy_ready.csv     <- primary; read by Step 11
    openbtai_breast_only_lesion_level_proxy_ready.csv
    openbtai_clinical_with_receptor_proxy_labels.csv
    openbtai_patient_level_external_proxy_ready.csv
    openbtai_feature_dictionary.json                       <- read by Step 11
  bcbm_project/metadata/
    step10_openbtai_build_summary.json
  bcbm_project/reports/tables/
    step10_openbtai_proxy_label_distribution.csv
  bcbm_project/cache/external_mapping/
    openbtai_merge_manifest.json

Run
---
python src/external/step_10_build_openbtai_external_proxy_dataset.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Paths  (resolved relative to cwd = project root, as set by run_pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("./bcbm_project").resolve()
STEP_NAME = "step_10_build_openbtai_external_proxy_dataset"

EXTERNAL_READY_DIR = PROJECT_ROOT / "data" / "external_ready"
METADATA_DIR = PROJECT_ROOT / "metadata"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
CACHE_DIR = PROJECT_ROOT / "cache" / "external_mapping"
LOGS_DIR = PROJECT_ROOT / "logs"


# ─────────────────────────────────────────────────────────────────────────────
# Maps
# ─────────────────────────────────────────────────────────────────────────────
PRIMARY_TUMOR_MAP: Dict[int, str] = {
    1: "Breast", 2: "NSCLC", 3: "SCLC", 4: "Melanoma", 5: "Kidney",
    6: "Colon", 7: "Bladder", 8: "Unknown", 9: "Ovary", 10: "Pharynx",
    11: "Bone sarcoma", 12: "Uterus",
}

BREAST_SUBTYPE_MAP: Dict[int, str] = {
    1: "Luminal A", 2: "Luminal B", 3: "Triple Negative", 4: "HER2-enriched",
}

PROXY_RULES: Dict[str, Dict[str, Any]] = {
    "Luminal A":       {"ER": 1, "PR": 1, "HER2": 0, "TNBC": 0},
    "Luminal B":       {"ER": 1, "PR": 1, "HER2": 0, "TNBC": 0},
    "Triple Negative": {"ER": 0, "PR": 0, "HER2": 0, "TNBC": 1},
    "HER2-enriched":   {"ER": None, "PR": None, "HER2": 1, "TNBC": 0},
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cache_manifest(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def is_cache_valid(manifest: Optional[Dict[str, Any]], input_hashes: Dict[str, str]) -> bool:
    if manifest is None:
        return False
    stored = manifest.get("input_file_sha256", {})
    return all(stored.get(k) == v for k, v in input_hashes.items())


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def setup_logger(log_file: Path) -> logging.Logger:
    ensure_dir(log_file.parent)
    logger = logging.getLogger(STEP_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    sh = logging.StreamHandler()
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def safe_int(x: Any) -> Optional[int]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:
        m = re.search(r"-?\d+", s)
        return int(m.group(0)) if m else None


def normalize_binary_marker(x: Any) -> Optional[int]:
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if not s or s in {"nan", "na", "n/a", "unknown", "unk", "missing"}:
        return None
    if s in {"1", "1.0", "yes", "y", "positive", "pos", "+", "true", "present"}:
        return 1
    if s in {"0", "0.0", "no", "n", "negative", "neg", "-", "false", "absent"}:
        return 0
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Proxy label derivation
# ─────────────────────────────────────────────────────────────────────────────
def add_proxy_labels_from_subtype(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ER/PR/HER2/TNBC proxy labels from the 4-class breast subtype code.

    IMPORTANT — these are PROXY labels, not direct receptor assays. The subtype
    code (1=Luminal A, 2=Luminal B, 3=Triple Negative, 4=HER2-enriched) cannot
    distinguish all true receptor states:
      - Luminal B (code 2) is mapped to HER2=0, but clinically Luminal B is
        frequently HER2-positive. The 4-class code carries no HER2 split within
        the luminal categories, so HER2+ Luminal B patients are mislabeled HER2=0
        here. This is a known limitation of subtype-derived labels.
      - HER2-enriched (code 4) has ER/PR set to None (unknown), not 0.
    Use these labels for apply-once external proxy validation only; treat the
    HER2 column with extra caution. Direct-assay columns (when present in the
    sheet) are loaded separately as `direct_*_bin` and are preferred when used.
    """
    d = df.copy()

    def er_proxy(code: Any) -> Optional[int]:
        code = safe_int(code)
        if code in (1, 2): return 1
        if code == 3:       return 0
        return None

    def pr_proxy(code: Any) -> Optional[int]:
        code = safe_int(code)
        if code in (1, 2): return 1
        if code == 3:       return 0
        return None

    def her2_proxy(code: Any) -> Optional[int]:
        code = safe_int(code)
        if code == 4:         return 1
        if code in (1, 2, 3): return 0
        return None

    def tnbc_proxy(code: Any) -> Optional[int]:
        code = safe_int(code)
        if code == 3:         return 1
        if code in (1, 2, 4): return 0
        return None

    d["external_er_proxy"]   = d["breast_subtype_code"].apply(er_proxy)
    d["external_pr_proxy"]   = d["breast_subtype_code"].apply(pr_proxy)
    d["external_her2_proxy"] = d["breast_subtype_code"].apply(her2_proxy)
    d["external_tnbc_proxy"] = d["breast_subtype_code"].apply(tnbc_proxy)

    non_breast = d["primary_tumor_code"] != 1
    for col in ["external_er_proxy", "external_pr_proxy", "external_her2_proxy", "external_tnbc_proxy"]:
        d.loc[non_breast, col] = np.nan

    return d


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────
def load_clinical_openbtai(path: Path, logger: logging.Logger) -> pd.DataFrame:
    logger.info("Loading clinical file: %s", path)
    raw = pd.read_excel(path, sheet_name=0, header=None)

    # The loader relies on a hardcoded layout: 15 preamble rows are dropped and
    # columns 0-6 / 15-17 carry specific fields. Validate the sheet shape up front
    # so a changed layout fails loudly here instead of silently producing garbage
    # labels downstream.
    if raw.shape[1] < 7:
        raise ValueError(
            f"OpenBTAI clinical sheet has {raw.shape[1]} columns; expected >= 7 "
            f"(patient_id..breast_subtype_code at column indices 0-6). The hardcoded "
            f"layout in load_clinical_openbtai may be stale for: {path}"
        )
    if raw.shape[0] <= 15:
        raise ValueError(
            f"OpenBTAI clinical sheet has {raw.shape[0]} rows; expected > 15 "
            f"(first 15 rows are dropped as preamble). File: {path}"
        )

    data = raw.iloc[15:].copy()
    out = pd.DataFrame({
        "patient_id_raw":       data.iloc[:, 0],
        "age_at_rm1":           data.iloc[:, 1],
        "sex_code":             data.iloc[:, 2],
        "gpa":                  data.iloc[:, 3],
        "clinical_lesion_id":   data.iloc[:, 4],
        "primary_tumor_code":   data.iloc[:, 5],
        "breast_subtype_code":  data.iloc[:, 6],
    })

    if raw.shape[1] > 17:
        out["direct_re_raw"]    = data.iloc[:, 15]
        out["direct_rp_raw"]    = data.iloc[:, 16]
        out["direct_her2_raw"]  = data.iloc[:, 17]
        out["direct_er_bin"]    = out["direct_re_raw"].apply(normalize_binary_marker)
        out["direct_pr_bin"]    = out["direct_rp_raw"].apply(normalize_binary_marker)
        out["direct_her2_bin"]  = out["direct_her2_raw"].apply(normalize_binary_marker)

    out["patient_id"]       = out["patient_id_raw"].apply(safe_int)
    out["patient_id_str"]   = out["patient_id"].apply(lambda x: f"{int(x):05d}" if pd.notna(x) else None)
    out["clinical_lesion_id"]   = out["clinical_lesion_id"].apply(safe_int)
    out["primary_tumor_code"]   = out["primary_tumor_code"].apply(safe_int)
    out["breast_subtype_code"]  = out["breast_subtype_code"].apply(safe_int)
    out["primary_tumor_name"]   = out["primary_tumor_code"].map(PRIMARY_TUMOR_MAP)
    out["breast_subtype_name"]  = out["breast_subtype_code"].map(BREAST_SUBTYPE_MAP)

    if int(out["patient_id"].notna().sum()) == 0:
        raise ValueError(
            "No valid integer patient IDs parsed from column 0 after dropping the 15 "
            f"preamble rows — the expected layout (data starting at row 15) appears wrong "
            f"for: {path}"
        )

    out = out[out["patient_id"].notna()].copy()
    out["patient_id"] = out["patient_id"].astype(int)
    out["sex"] = out["sex_code"].map({1: "Female", 2: "Male"})
    out = add_proxy_labels_from_subtype(out)

    logger.info("  Clinical rows loaded: %d  |  unique patients: %d",
                len(out), out["patient_id"].nunique())
    return out.reset_index(drop=True)


def load_radiomics_openbtai(path: Path, logger: logging.Logger) -> pd.DataFrame:
    logger.info("Loading radiomics file: %s", path)
    rad = pd.read_excel(path, sheet_name=0)
    rad = rad.rename(columns={
        "Patient": "patient_id", "Timepoint": "timepoint",
        "Lesion": "lesion_id", "Label": "segment_label",
        "Segment": "segment_name", "Image": "image_file", "Mask": "mask_file",
    })
    for col in ["patient_id", "timepoint", "lesion_id", "segment_label"]:
        if col in rad.columns:
            rad[col] = pd.to_numeric(rad[col], errors="coerce").astype("Int64")
    logger.info("  Radiomics rows: %d  |  unique patients: %d",
                len(rad), rad["patient_id"].nunique())
    return rad


def load_morphology_openbtai(path: Path, logger: logging.Logger) -> pd.DataFrame:
    logger.info("Loading morphology file: %s", path)
    morph = pd.read_excel(path, sheet_name=0)
    morph = morph.rename(columns={
        "PATIENT": "patient_id", "TIME POINT": "timepoint", "LESION": "lesion_id",
    })
    for col in ["patient_id", "timepoint", "lesion_id"]:
        if col in morph.columns:
            morph[col] = pd.to_numeric(morph[col], errors="coerce").astype("Int64")
    key_cols = {"patient_id", "timepoint", "lesion_id"}
    morph = morph.rename(
        columns={c: f"morph_{c.lower()}" for c in morph.columns if c not in key_cols}
    )
    logger.info("  Morphology rows: %d  |  unique patients: %d",
                len(morph), morph["patient_id"].nunique())
    return morph


# ─────────────────────────────────────────────────────────────────────────────
# Feature inference
# ─────────────────────────────────────────────────────────────────────────────
def infer_radiomics_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {
        "patient_id", "timepoint", "segment_label", "lesion_id", "segment_name",
        "image_file", "mask_file", "clinical_lesion_id", "patient_id_raw",
        "patient_id_str", "primary_tumor_name", "breast_subtype_name", "sex",
        "external_er_proxy", "external_pr_proxy", "external_her2_proxy", "external_tnbc_proxy",
        "direct_er_bin", "direct_pr_bin", "direct_her2_bin",
        "direct_re_raw", "direct_rp_raw", "direct_her2_raw",
    }
    return [
        c for c in df.columns
        if c not in exclude
        and not c.startswith("diagnostics_")
        and not c.startswith("morph_")
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def build_feature_dictionary(lesion_df: pd.DataFrame, rad_feature_cols: List[str]) -> Dict[str, Any]:
    morph_cols = [c for c in lesion_df.columns if c.startswith("morph_")
                  and pd.api.types.is_numeric_dtype(lesion_df[c])]
    proxy_label_cols = [
        "external_er_proxy", "external_pr_proxy", "external_her2_proxy", "external_tnbc_proxy",
    ]
    direct_label_cols = [
        c for c in ["direct_er_bin", "direct_pr_bin", "direct_her2_bin"]
        if c in lesion_df.columns
    ]

    # patient-level feature names after aggregation (mean/std/min/max)
    def agg_names(cols: List[str]) -> List[str]:
        return [f"ext_{c}_{agg}" for c in cols for agg in ("mean", "std", "min", "max")]

    return {
        "step": STEP_NAME,
        "created_at": datetime.utcnow().isoformat(),
        "lesion_level": {
            "radiomics_features": rad_feature_cols,
            "morphology_features": morph_cols,
        },
        "patient_level_aggregated": {
            "radiomics_features": agg_names(rad_feature_cols),
            "morphology_features": agg_names(morph_cols),
        },
        "proxy_label_columns": proxy_label_cols,
        "direct_label_columns": direct_label_cols,
        "n_radiomics_features": len(rad_feature_cols),
        "n_morphology_features": len(morph_cols),
        "proxy_rules": PROXY_RULES,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Patient-level aggregation
# ─────────────────────────────────────────────────────────────────────────────
def build_patient_level_table(lesion_df: pd.DataFrame) -> pd.DataFrame:
    d = lesion_df.copy()
    feature_cols = [
        c for c in d.columns
        if pd.api.types.is_numeric_dtype(d[c])
        and c not in {
            "patient_id", "timepoint", "lesion_id", "segment_label",
            "clinical_lesion_id", "primary_tumor_code", "breast_subtype_code",
            "sex_code", "age_at_rm1", "gpa",
            "external_er_proxy", "external_pr_proxy", "external_her2_proxy", "external_tnbc_proxy",
            "direct_er_bin", "direct_pr_bin", "direct_her2_bin",
        }
    ]

    agg = d.groupby("patient_id")[feature_cols].agg(["mean", "std", "min", "max"])
    agg.columns = [f"ext_{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()

    label_cols = [
        "primary_tumor_code", "primary_tumor_name", "breast_subtype_code", "breast_subtype_name",
        "external_er_proxy", "external_pr_proxy", "external_her2_proxy", "external_tnbc_proxy",
        "direct_er_bin", "direct_pr_bin", "direct_her2_bin",
        "age_at_rm1", "sex_code", "sex", "gpa",
    ]
    existing = [c for c in label_cols if c in d.columns]
    labels = d.groupby("patient_id")[existing].first().reset_index()

    counts = d.groupby("patient_id").agg(
        n_rows=("patient_id", "size"),
        n_timepoints=("timepoint", "nunique"),
        n_lesions=("lesion_id", "nunique"),
    ).reset_index()

    out = labels.merge(counts, on="patient_id", how="left").merge(agg, on="patient_id", how="left")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Step 10 — Build OpenBTAI external proxy dataset")
    parser.add_argument("--clinical",   type=Path, default=Path("OpenBTAI_METS_ClinicalData_Nov2023.xlsx"))
    parser.add_argument("--radiomics",  type=Path, default=Path("OpenBTAI_RADIOMICS.xlsx"))
    parser.add_argument("--morphology", type=Path, default=Path("OpenBTAI_MORPHOLOGICAL_MEASUREMENTS.xlsx"))
    parser.add_argument("--force",    action="store_true", help="Re-run even if cache is valid")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache check (same as --force)")
    args = parser.parse_args()

    # Ensure output directories exist
    for d in [EXTERNAL_READY_DIR, METADATA_DIR, TABLES_DIR, CACHE_DIR, LOGS_DIR]:
        ensure_dir(d)

    logger = setup_logger(LOGS_DIR / f"{STEP_NAME}.log")
    logger.info("Step 10 started — Build OpenBTAI External Proxy Dataset")
    logger.info("PROJECT_ROOT: %s", PROJECT_ROOT)
    logger.info("Output dir:   %s", EXTERNAL_READY_DIR)

    # ── Cache check ───────────────────────────────────────────────────────────
    manifest_path = CACHE_DIR / "openbtai_merge_manifest.json"
    input_hashes: Dict[str, str] = {}
    for key, path in [("clinical", args.clinical), ("radiomics", args.radiomics), ("morphology", args.morphology)]:
        if Path(path).exists():
            input_hashes[key] = compute_file_sha256(Path(path))
    cached = load_cache_manifest(manifest_path)
    if not args.force and not args.no_cache and is_cache_valid(cached, input_hashes):
        logger.info(
            "Cache valid — all input file hashes unchanged. Skipping Step 10. "
            "Run with --force to rebuild."
        )
        return

    # ── Load ──────────────────────────────────────────────────────────────────
    clinical  = load_clinical_openbtai(args.clinical, logger)
    radiomics = load_radiomics_openbtai(args.radiomics, logger)
    morphology = load_morphology_openbtai(args.morphology, logger)

    # ── Merge ─────────────────────────────────────────────────────────────────
    logger.info("Merging radiomics + clinical on patient_id / lesion_id ...")
    merged = radiomics.merge(
        clinical,
        left_on=["patient_id", "lesion_id"],
        right_on=["patient_id", "clinical_lesion_id"],
        how="left",
        validate="many_to_one",
    )
    logger.info("Merging morphology on patient_id / timepoint / lesion_id ...")
    merged = merged.merge(
        morphology,
        on=["patient_id", "timepoint", "lesion_id"],
        how="left",
        validate="many_to_one",
    )

    merged["is_breast_primary"]   = merged["primary_tumor_code"].eq(1)
    merged["has_any_proxy_label"] = merged[[
        "external_er_proxy", "external_pr_proxy", "external_her2_proxy", "external_tnbc_proxy"
    ]].notna().any(axis=1)

    # ── Aggregation ───────────────────────────────────────────────────────────
    logger.info("Building patient-level table ...")
    patient_level = build_patient_level_table(merged)

    breast_lesion   = merged[merged["is_breast_primary"]].copy()
    breast_patient  = patient_level[patient_level["primary_tumor_code"].eq(1)].copy()

    logger.info("  All patients: %d  |  Breast patients: %d",
                patient_level["patient_id"].nunique(),
                breast_patient["patient_id"].nunique())

    # ── Feature dictionary ────────────────────────────────────────────────────
    rad_feature_cols = infer_radiomics_feature_columns(merged)
    feature_dict     = build_feature_dictionary(merged, rad_feature_cols)

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    csv_saves = {
        "openbtai_clinical_with_receptor_proxy_labels.csv":     clinical,
        "openbtai_lesion_level_radiomics_morphology_proxy_labels.csv": merged,
        "openbtai_patient_level_external_proxy_ready.csv":      patient_level,
        "openbtai_breast_only_lesion_level_proxy_ready.csv":    breast_lesion,
        "openbtai_breast_only_patient_level_proxy_ready.csv":   breast_patient,
    }
    for fname, df in csv_saves.items():
        path = EXTERNAL_READY_DIR / fname
        df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info("  Saved: %s  (%d rows)", path.name, len(df))

    # ── Feature dictionary ────────────────────────────────────────────────────
    feat_dict_path = EXTERNAL_READY_DIR / "openbtai_feature_dictionary.json"
    save_json(feature_dict, feat_dict_path)
    logger.info("  Feature dictionary saved: %s", feat_dict_path.name)

    # ── Label distribution report ─────────────────────────────────────────────
    proxy_cols = ["external_er_proxy", "external_pr_proxy", "external_her2_proxy", "external_tnbc_proxy"]
    dist_rows = []
    for col in proxy_cols:
        if col not in breast_patient.columns:
            continue
        vc = breast_patient[col].value_counts(dropna=False)
        dist_rows.append({
            "label": col,
            "n_positive":  int(vc.get(1, 0)),
            "n_negative":  int(vc.get(0, 0)),
            "n_missing":   int(breast_patient[col].isna().sum()),
            "n_total":     int(len(breast_patient)),
        })
    if dist_rows:
        dist_df = pd.DataFrame(dist_rows)
        dist_path = TABLES_DIR / "step10_openbtai_proxy_label_distribution.csv"
        dist_df.to_csv(dist_path, index=False, encoding="utf-8-sig")
        logger.info("  Label distribution saved: %s", dist_path.name)

    # ── Feature mapping parquet (cache for Step 11) ───────────────────────────
    feat_map_df = pd.DataFrame({
        "feature_name": rad_feature_cols,
        "aggregated_names": [f"ext_{c}_mean | ext_{c}_std | ext_{c}_min | ext_{c}_max" for c in rad_feature_cols],
    })
    feat_map_df.to_parquet(CACHE_DIR / "openbtai_feature_mapping.parquet", index=False)
    logger.info("  Feature mapping parquet saved: %d radiomics features", len(rad_feature_cols))

    # ── Cache manifest ────────────────────────────────────────────────────────
    manifest = {
        "step": STEP_NAME,
        "created_at": datetime.utcnow().isoformat(),
        "status": "valid",
        "input_files": {
            "clinical":   str(args.clinical),
            "radiomics":  str(args.radiomics),
            "morphology": str(args.morphology),
        },
        "input_file_sha256": input_hashes,
        "row_counts": {
            "clinical":       int(len(clinical)),
            "radiomics":      int(len(radiomics)),
            "morphology":     int(len(morphology)),
            "merged_lesion":  int(len(merged)),
            "patient_level":  int(len(patient_level)),
            "breast_lesion":  int(len(breast_lesion)),
            "breast_patient": int(len(breast_patient)),
        },
    }
    save_json(manifest, manifest_path)

    # ── Step summary ──────────────────────────────────────────────────────────
    summary = {
        "step": STEP_NAME,
        "completed_at": datetime.utcnow().isoformat(),
        "input_files": manifest["input_files"],
        "row_counts": manifest["row_counts"],
        "patients": {
            "all":    int(merged["patient_id"].nunique()),
            "breast": int(breast_lesion["patient_id"].nunique()),
        },
        "primary_tumor_counts": patient_level["primary_tumor_name"].value_counts(dropna=False).to_dict()
                                 if "primary_tumor_name" in patient_level.columns else {},
        "breast_subtype_counts": breast_patient["breast_subtype_name"].value_counts(dropna=False).to_dict()
                                  if "breast_subtype_name" in breast_patient.columns else {},
        "proxy_label_coverage_breast": {
            col: int(breast_patient[col].notna().sum())
            for col in proxy_cols if col in breast_patient.columns
        },
        "direct_receptor_coverage_breast": {
            col: int(breast_patient[col].notna().sum())
            for col in ["direct_er_bin", "direct_pr_bin", "direct_her2_bin"]
            if col in breast_patient.columns
        },
        "n_radiomics_features":  len(rad_feature_cols),
        "outputs": {
            "external_ready_dir": str(EXTERNAL_READY_DIR),
            "primary_file":       str(EXTERNAL_READY_DIR / "openbtai_breast_only_patient_level_proxy_ready.csv"),
            "feature_dictionary": str(feat_dict_path),
            "summary_json":       str(METADATA_DIR / "step10_openbtai_build_summary.json"),
        },
    }
    save_json(summary, METADATA_DIR / "step10_openbtai_build_summary.json")
    logger.info("Step 10 complete.  Breast patients: %d  |  Radiomics features: %d",
                summary["patients"]["breast"], summary["n_radiomics_features"])


if __name__ == "__main__":
    main()
