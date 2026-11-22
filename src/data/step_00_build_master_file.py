from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# Project rule: use a logger, never print() in step scripts. Step 00 runs before
# Step 01 creates the project tree, so it logs to the console only.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("step_00_build_master_file")


# =========================================================
# SETTINGS
# =========================================================
RADIOMICS_XLSX = Path("BCBM-RadioGenomic_Radiomics_Data.xlsx")
CLINICAL_XLSX = Path("BCBM-RadioGenomics-Clinical-data.xlsx")
MAPPING_XLSX = Path("real_files_mapping.xlsx")

RADIOMICS_SHEET = "merged_orig"
CLINICAL_SHEET = "Clinical+Genetics"
MAPPING_SHEET = "Sheet1"

# All outputs go into bcbm_project/ (created on demand).
# Step 01 reads BCBM_Full_Data.xlsx from bcbm_project/data/raw/.
_PROJECT = Path("./bcbm_project")
OUTPUT_MERGED_XLSX    = _PROJECT / "data" / "raw"     / "BCBM_Full_Data.xlsx"
OUTPUT_MISSING_CSV    = _PROJECT / "reports" / "tables" / "step00_missing_report.csv"
OUTPUT_DUPLICATES_CSV = _PROJECT / "reports" / "tables" / "step00_duplicate_key_report.csv"
OUTPUT_AUDIT_JSON     = _PROJECT / "metadata"          / "step00_master_merge_audit.json"


# =========================================================
# HELPERS
# =========================================================
def norm_col(col: str) -> str:
    col = str(col).strip()
    col = re.sub(r"\s+", " ", col)
    return col


def norm_text(s: object) -> str:
    s = "" if s is None else str(s).strip()
    s = s.replace("\\", "/")
    s = re.sub(r"\.nii\.gz$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\.nii$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return s


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [norm_col(c) for c in df.columns]
    return df


def extract_segmentation_name(mask_filename: object) -> str:
    """
    Convert a mapping filename like:
        BCBM-RadioGenomics-155-4_mask_Plan1_TgtA_220Gy80.nii.gz
    into:
        mask_Plan1_TgtA_220Gy80

    This is designed to match radiomics Segmentation_Name values.
    """
    name = "" if mask_filename is None else str(mask_filename).strip()
    name = re.sub(r"\.nii(\.gz)?$", "", name, flags=re.IGNORECASE)

    m = re.search(r"(_mask_.+)$", name, flags=re.IGNORECASE)
    if m:
        return m.group(1).lstrip("_")  # -> mask_Plan1_TgtA_220Gy80

    return name


def choose_best_image(group: pd.DataFrame) -> str:
    vals = group["image_filename"].dropna().astype(str).str.strip().unique().tolist()
    if not vals:
        return ""

    preferred = [v for v in vals if "_image_ss_n4.nii.gz" in v.lower()]
    return preferred[0] if preferred else vals[0]


def summarize_duplicate_keys(df: pd.DataFrame, key_cols: List[str], source_name: str) -> pd.DataFrame:
    """
    Return one row per duplicated key group with counts and sample values.
    """
    if df.empty:
        return pd.DataFrame(columns=key_cols + ["row_count", "source"])

    counts = (
        df.groupby(key_cols, dropna=False)
        .size()
        .reset_index(name="row_count")
    )

    dup = counts[counts["row_count"] > 1].copy()
    if dup.empty:
        dup["source"] = source_name
        return dup

    dup["source"] = source_name
    dup = dup.sort_values(["row_count"] + key_cols, ascending=[False] + [True] * len(key_cols))
    return dup


# =========================================================
# LOAD FILES
# =========================================================
def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rad = pd.read_excel(RADIOMICS_XLSX, sheet_name=RADIOMICS_SHEET)
    clin = pd.read_excel(CLINICAL_XLSX, sheet_name=CLINICAL_SHEET)
    mapping = pd.read_excel(MAPPING_XLSX, sheet_name=MAPPING_SHEET)

    rad = clean_columns(rad)
    clin = clean_columns(clin)
    mapping = clean_columns(mapping)

    return rad, clin, mapping


# =========================================================
# PREP INPUTS
# =========================================================
def prepare_clinical(clin: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "ID": "case_id",
        "Age": "Age",
        "Year": "Year",
        "Pixel Spacing": "Pixel_Spacing",
        "Manufacturer": "Manufacturer",
        "Magnetic Field Strength ID": "Magnetic_Field_Strength_ID",
        "Sex": "Sex",
        "ER": "ER",
        "PR": "PR",
        "HER2": "HER2",
    }

    actual_map = {}
    for c in clin.columns:
        c_simple = c.strip()
        if c_simple in rename_map:
            actual_map[c] = rename_map[c_simple]

    clin = clin.rename(columns=actual_map).copy()

    required = [
        "case_id", "Age", "Year", "Pixel_Spacing", "Manufacturer",
        "Magnetic_Field_Strength_ID", "Sex", "ER", "PR", "HER2"
    ]
    missing = [c for c in required if c not in clin.columns]
    if missing:
        raise ValueError(f"Clinical file is missing required columns after cleaning: {missing}")

    clin["case_id"] = clin["case_id"].astype(str).str.strip()
    clin["patient_base"] = clin["case_id"].str.extract(r"BCBM-RadioGenomics-(\d+)-(\d+)")[0]
    clin["visit_index"] = clin["case_id"].str.extract(r"BCBM-RadioGenomics-(\d+)-(\d+)")[1]

    return clin


def prepare_radiomics(rad: pd.DataFrame) -> pd.DataFrame:
    required = ["FilenamePrefix", "Segmentation_Name"]
    missing = [c for c in required if c not in rad.columns]
    if missing:
        raise ValueError(f"Radiomics file is missing required columns: {missing}")

    rad = rad.copy()
    rad["FilenamePrefix"] = rad["FilenamePrefix"].astype(str).str.strip()
    rad["Segmentation_Name"] = rad["Segmentation_Name"].astype(str).str.strip()
    rad["patient_base"] = rad["FilenamePrefix"].str.extract(r"BCBM-RadioGenomics-(\d+)-(\d+)")[0]
    rad["visit_index"] = rad["FilenamePrefix"].str.extract(r"BCBM-RadioGenomics-(\d+)-(\d+)")[1]
    rad["seg_norm"] = rad["Segmentation_Name"].apply(norm_text)

    return rad


def prepare_mapping(mapping: pd.DataFrame) -> pd.DataFrame:
    required = ["patient_folder", "image_filename", "mask_filename"]
    missing = [c for c in required if c not in mapping.columns]
    if missing:
        raise ValueError(f"Mapping file is missing required columns: {missing}")

    mapping = mapping.copy()
    for c in required:
        mapping[c] = mapping[c].astype(str).str.strip()

    mapping["FilenamePrefix"] = mapping["patient_folder"]
    mapping["Segmentation_Name"] = mapping["mask_filename"].apply(extract_segmentation_name)
    mapping["seg_norm"] = mapping["Segmentation_Name"].apply(norm_text)

    mapping["image_path"] = (
        "BCBM-RadioGenomics_Images_Masks_Dec2024\\"
        + mapping["patient_folder"] + "\\"
        + mapping["image_filename"]
    )
    mapping["mask_path"] = (
        "BCBM-RadioGenomics_Images_Masks_Dec2024\\"
        + mapping["patient_folder"] + "\\"
        + mapping["mask_filename"]
    )
    mapping["matched_mask_file"] = mapping["mask_filename"].str.lower()

    best_image = (
        mapping.groupby("FilenamePrefix", as_index=False)
        .apply(lambda g: pd.Series({"image_filename_best": choose_best_image(g)}), include_groups=False)
        .reset_index(drop=True)
    )

    mapping = mapping.merge(best_image, on="FilenamePrefix", how="left", validate="many_to_one")

    mapping["image_path"] = (
        "BCBM-RadioGenomics_Images_Masks_Dec2024\\"
        + mapping["patient_folder"] + "\\"
        + mapping["image_filename_best"]
    )

    # Keep one row per exact case + segmentation
    mapping = mapping.drop_duplicates(subset=["FilenamePrefix", "seg_norm"]).copy()

    return mapping


# =========================================================
# AUDIT HELPERS
# =========================================================
def build_missing_report(merged: pd.DataFrame) -> pd.DataFrame:
    missing = pd.DataFrame({
        "FilenamePrefix": merged["FilenamePrefix"],
        "Segmentation_Name": merged["Segmentation_Name"],
        "missing_mapping": merged["mask_path"].isna() | (merged["mask_path"].astype(str).str.strip() == ""),
        "missing_clinical": merged["case_id"].isna() | (merged["case_id"].astype(str).str.strip() == ""),
    })

    missing["missing_any"] = missing["missing_mapping"] | missing["missing_clinical"]

    if not missing["missing_any"].any():
        return pd.DataFrame(columns=[
            "FilenamePrefix", "Segmentation_Name", "missing_mapping",
            "missing_clinical", "reason"
        ])

    report = missing.loc[missing["missing_any"]].copy()
    reasons = []
    for _, row in report.iterrows():
        row_reasons = []
        if row["missing_mapping"]:
            row_reasons.append("missing_image_or_mask_mapping")
        if row["missing_clinical"]:
            row_reasons.append("missing_clinical_case")
        reasons.append(" | ".join(row_reasons))

    report["reason"] = reasons
    return report


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    rad_raw, clin_raw, mapping_raw = load_inputs()

    rad = prepare_radiomics(rad_raw)
    clin = prepare_clinical(clin_raw)
    mapping = prepare_mapping(mapping_raw)

    # -----------------------------------------------------
    # Duplicate audits before merge
    # -----------------------------------------------------
    rad_dup = summarize_duplicate_keys(
        rad,
        key_cols=["FilenamePrefix", "seg_norm"],
        source_name="radiomics"
    )

    mapping_dup_before_drop = prepare_mapping_for_dup_audit(mapping_raw)

    mapping_dup = summarize_duplicate_keys(
        mapping_dup_before_drop,
        key_cols=["FilenamePrefix", "seg_norm"],
        source_name="mapping_before_dedup"
    )

    clin_dup = summarize_duplicate_keys(
        clin,
        key_cols=["case_id"],
        source_name="clinical"
    )

    duplicate_report = pd.concat(
        [rad_dup, mapping_dup, clin_dup],
        ignore_index=True,
        sort=False
    )

    # -----------------------------------------------------
    # Merge radiomics + mapping
    # many radiomics rows can point to one mapping row
    # -----------------------------------------------------
    merged = rad.merge(
        mapping[["FilenamePrefix", "seg_norm", "image_path", "mask_path", "matched_mask_file", "mask_filename"]],
        on=["FilenamePrefix", "seg_norm"],
        how="left",
        validate="many_to_one",
    )

    # -----------------------------------------------------
    # Merge clinical
    # many radiomics rows -> one clinical case row
    # -----------------------------------------------------
    merged = merged.merge(
        clin,
        left_on="FilenamePrefix",
        right_on="case_id",
        how="left",
        validate="many_to_one",
    )

    # -----------------------------------------------------
    # Reorder columns
    # -----------------------------------------------------
    first_cols = [
        "FilenamePrefix", "Segmentation_Name", "case_id",
        "patient_base_x", "visit_index_x",
        "Age", "Year", "Pixel_Spacing", "Manufacturer",
        "Magnetic_Field_Strength_ID", "Sex", "ER", "PR", "HER2",
        "image_path", "mask_path", "matched_mask_file", "mask_filename",
    ]
    first_cols = [c for c in first_cols if c in merged.columns]
    other_cols = [c for c in merged.columns if c not in first_cols]
    merged = merged[first_cols + other_cols].copy()

    rename_out = {}
    if "patient_base_x" in merged.columns:
        rename_out["patient_base_x"] = "patient_base"
    if "visit_index_x" in merged.columns:
        rename_out["visit_index_x"] = "visit_index"
    merged = merged.rename(columns=rename_out)

    drop_cols = [c for c in ["patient_base_y", "visit_index_y", "seg_norm"] if c in merged.columns]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)

    # -----------------------------------------------------
    # Missing report
    # -----------------------------------------------------
    missing_report = build_missing_report(merged)

    # -----------------------------------------------------
    # Save outputs
    # -----------------------------------------------------
    for _p in [OUTPUT_MERGED_XLSX, OUTPUT_MISSING_CSV, OUTPUT_DUPLICATES_CSV, OUTPUT_AUDIT_JSON]:
        _p.parent.mkdir(parents=True, exist_ok=True)

    merged.to_excel(OUTPUT_MERGED_XLSX, index=False)
    missing_report.to_csv(OUTPUT_MISSING_CSV, index=False, encoding="utf-8-sig")
    duplicate_report.to_csv(OUTPUT_DUPLICATES_CSV, index=False, encoding="utf-8-sig")

    audit = {
        "radiomics_rows": int(rad.shape[0]),
        "clinical_rows": int(clin.shape[0]),
        "mapping_rows_after_dedup": int(mapping.shape[0]),
        "final_rows": int(merged.shape[0]),
        "final_columns": int(merged.shape[1]),
        "radiomics_duplicate_key_groups": int(rad_dup.shape[0]),
        "mapping_duplicate_key_groups_before_dedup": int(mapping_dup.shape[0]),
        "clinical_duplicate_case_id_groups": int(clin_dup.shape[0]),
        "missing_mapping_rows": int(missing_report["missing_mapping"].sum()) if not missing_report.empty else 0,
        "missing_clinical_rows": int(missing_report["missing_clinical"].sum()) if not missing_report.empty else 0,
        "missing_any_rows": int(missing_report.shape[0]),
        "notes": [
            "Final file is radiomics-driven: one row per radiomics row.",
            "Radiomics-to-mapping merge uses many_to_one because duplicate radiomics keys can exist.",
            "Clinical data is merged by case using FilenamePrefix == case_id.",
            "Mapping segmentation names are reconstructed from mask filenames to match radiomics Segmentation_Name.",
        ],
    }

    OUTPUT_AUDIT_JSON.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Saved: %s", OUTPUT_MERGED_XLSX)
    logger.info("Saved: %s", OUTPUT_MISSING_CSV)
    logger.info("Saved: %s", OUTPUT_DUPLICATES_CSV)
    logger.info("Saved: %s", OUTPUT_AUDIT_JSON)
    logger.info("Master merge audit:\n%s", json.dumps(audit, indent=2))


def prepare_mapping_for_dup_audit(mapping_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build mapping keys before dedup, so duplicate groups can be audited.
    """
    mapping = mapping_raw.copy()
    mapping = clean_columns(mapping)

    required = ["patient_folder", "image_filename", "mask_filename"]
    missing = [c for c in required if c not in mapping.columns]
    if missing:
        raise ValueError(f"Mapping file is missing required columns: {missing}")

    for c in required:
        mapping[c] = mapping[c].astype(str).str.strip()

    mapping["FilenamePrefix"] = mapping["patient_folder"]
    mapping["Segmentation_Name"] = mapping["mask_filename"].apply(extract_segmentation_name)
    mapping["seg_norm"] = mapping["Segmentation_Name"].apply(norm_text)

    return mapping


if __name__ == "__main__":
    main()