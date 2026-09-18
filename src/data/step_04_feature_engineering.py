from __future__ import annotations

"""
Step 04 — Patient-Level Feature Engineering and Modeling Preparation (Scanner-Aware Final)
==========================================================================================
This revision keeps the original Stage 04 functionality and adds explicit handling for
scanner / field-strength confounding.

Added in this revision
----------------------
FIX-11  Explicit magnetic-field-strength harmonization helpers were added. The stage now
        canonicalizes field-strength metadata into a single numeric column:
        magnetic_field_strength_t.

FIX-12  Scanner-aware patient summaries are added, including dominant field strength,
        number of unique scanner strengths, and entropy of field-strength distribution.

FIX-13  Stratification keys can now optionally include dominant field strength so train /
        valid / test splits are less likely to drift across scanner domains.

FIX-14  A scanner confounding audit is produced, including label-vs-field-strength
        contingency summaries and radiomics shift screening by field strength.

FIX-15  Feature dictionary now records feature subsets for:
        - all numeric features
        - scanner covariates only
        - numeric features excluding scanner covariates
        - field-strength-only candidates
        This makes Stage 05 ablation studies straightforward.

FIX-16  Patient-level table now preserves scanner covariates explicitly using stable,
        discoverable names instead of only generic meta aggregates.

FIX-17  Stage summary now includes field-strength coverage and split balance by scanner.

Important note
--------------
This stage does NOT apply ComBat to avoid leakage at this point in the pipeline.
Instead, it prepares scanner-aware covariates and audit outputs so Stage 05 can run:
  1) radiomics only
  2) scanner only
  3) radiomics + scanner
Any future harmonization should be fit on TRAIN only and then applied to valid/test.
"""

import json
import argparse
import logging
import math
import os
import platform
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path("./bcbm_project").resolve()
STEP_NAME = "step_04_feature_engineering"

INPUT_LESION_ONLY = PROJECT_ROOT / "data" / "processed" / "analysis_ready_step03_lesion_only.csv"
INPUT_FULL = PROJECT_ROOT / "data" / "processed" / "analysis_ready_step03_full.csv"
STEP03_SUMMARY_PATH = PROJECT_ROOT / "metadata" / "step03_summary.json"

SAVE_PARQUET_IF_AVAILABLE = True
SAVE_LONG_CASE_TABLE = True
SAVE_LONG_PATIENT_TABLE = True
SAVE_SPLITS = True

PATIENT_ID_COL = "patient_base"
CASE_ID_COL = "case_id"
VISIT_ID_COL = "visit_index"
SEGMENTATION_COL = "Segmentation_Name"
MASK_CATEGORY_COL = "final_mask_category"
ANALYSIS_FLAG_COL = "is_analysis_candidate_final"

LABEL_COLUMNS = ["ER_bin", "PR_bin", "HER2_bin"]
RAW_LABEL_COLUMNS = ["ER", "PR", "HER2"]

OPTIONAL_METADATA_COLUMNS = [
    "Age", "Year", "Pixel Spacing", "Pixel_Spacing", "Manufacturer",
    "Magnetic Field Strength ID", "Magnetic_Field_Strength_ID",
    "Gender", "Sex", "mask_descriptor_canonical", "mask_descriptor_source",
    "mask_volume_cc", "mask_voxel_count", "n_components", "n_slices_involved",
    "bbox_dim_0", "bbox_dim_1", "bbox_dim_2", "mask_fill_ratio_in_bbox",
    "final_resolution_confidence", "final_resolution_source",
    "initial_mask_category",
]

MAX_PAIRWISE_FEATURES = 20
TOP_PATIENT_RAD_FEATURES = 20
# When True, every patient is summarized on the SAME radiomic features (all of them), instead of
# the 20 with the highest variance within that patient. The per-patient variant left ~71% of the
# patient-level radiomic matrix empty (each column populated only for the patients where it ranked
# top-20), which can fairly be read as an artefact-generating design.
FIXED_PATIENT_FEATURE_SET = True
# Sensitivity option: keep only the earliest eligible examination per
# patient, so that lesions imaged before and after treatment are not pooled under one
# patient-level receptor label. Enabled with --one-session; writes *_onesession outputs and
# never overwrites the primary tables.
ONE_SESSION_SUFFIX = "_onesession"
# Aggregation sensitivity. The primary table summarises each
# radiomic feature over a patient's lesions with six statistics (mean, SD, min, max, CV, IQR);
# SD, CV, IQR and the extremes all move with the number of lesions, which ranges from one to
# several hundred. "median" keeps one robust summary per feature; "largest" takes the feature
# from the patient's largest lesion by mask volume. Only the radiomic block changes; every other
# column is computed exactly as in the primary table. Writes *_agg_<mode> outputs.
RADIOMIC_AGGREGATIONS = ("six_stat", "median", "largest")
HIGH_MISSINGNESS_THRESHOLD = 0.95
EPS = 1e-12

TRAIN_FRACTION = 0.70
VALID_FRACTION = 0.15
TEST_FRACTION = 0.15

GLOBAL_SEED = 42
MIN_PATIENTS_PER_STRATUM = 3
INCLUDE_FIELD_STRENGTH_IN_STRATIFICATION = True
ENABLE_SCANNER_CONFOUNDING_AUDIT = True
FIELD_STRENGTH_NUMERIC_COL = "magnetic_field_strength_t"
FIELD_STRENGTH_LABEL_COL = "magnetic_field_strength_label"
FIELD_STRENGTH_DOMINANT_COL = "dominant_field_strength_t"
MANUFACTURER_DOMINANT_COL = "dominant_manufacturer"

RADIOMICS_PREFIXES = [
    "original_", "wavelet_", "log_sigma_", "lbp_",
    "shape_", "firstorder_", "glcm_", "glrlm_",
    "glszm_", "gldm_", "ngtdm_", "diagnostics_",
]

SCANNER_COVARIATE_PRIORITY = [
    FIELD_STRENGTH_DOMINANT_COL,
    "patient_meta_magnetic_field_strength_id_mode",
    FIELD_STRENGTH_LABEL_COL,
    MANUFACTURER_DOMINANT_COL,
    "patient_meta_manufacturer_mode",
    "n_unique_field_strength_patient",
    "entropy_magnetic_field_strength_patient",
    "n_unique_manufacturer_patient",
    "entropy_manufacturer_patient",
]


@dataclass
class Step04Config:
    project_root: str = str(PROJECT_ROOT)
    step_name: str = STEP_NAME
    input_lesion_only_csv: str = str(INPUT_LESION_ONLY)
    input_full_csv: str = str(INPUT_FULL)
    step03_summary_path: str = str(STEP03_SUMMARY_PATH)
    save_parquet_if_available: bool = SAVE_PARQUET_IF_AVAILABLE
    save_long_case_table: bool = SAVE_LONG_CASE_TABLE
    save_long_patient_table: bool = SAVE_LONG_PATIENT_TABLE
    save_splits: bool = SAVE_SPLITS
    patient_id_col: str = PATIENT_ID_COL
    case_id_col: str = CASE_ID_COL
    visit_id_col: str = VISIT_ID_COL
    segmentation_col: str = SEGMENTATION_COL
    mask_category_col: str = MASK_CATEGORY_COL
    analysis_flag_col: str = ANALYSIS_FLAG_COL
    label_columns: List[str] = None
    raw_label_columns: List[str] = None
    optional_metadata_columns: List[str] = None
    max_pairwise_features: int = MAX_PAIRWISE_FEATURES
    top_patient_rad_features: int = TOP_PATIENT_RAD_FEATURES
    fixed_patient_feature_set: bool = FIXED_PATIENT_FEATURE_SET
    one_session_per_patient: bool = False
    radiomic_aggregation: str = "six_stat"
    high_missingness_threshold: float = HIGH_MISSINGNESS_THRESHOLD
    train_fraction: float = TRAIN_FRACTION
    valid_fraction: float = VALID_FRACTION
    test_fraction: float = TEST_FRACTION
    global_seed: int = GLOBAL_SEED
    min_patients_per_stratum: int = MIN_PATIENTS_PER_STRATUM
    include_field_strength_in_stratification: bool = INCLUDE_FIELD_STRENGTH_IN_STRATIFICATION
    enable_scanner_confounding_audit: bool = ENABLE_SCANNER_CONFOUNDING_AUDIT
    field_strength_numeric_col: str = FIELD_STRENGTH_NUMERIC_COL
    field_strength_label_col: str = FIELD_STRENGTH_LABEL_COL
    field_strength_dominant_col: str = FIELD_STRENGTH_DOMINANT_COL
    manufacturer_dominant_col: str = MANUFACTURER_DOMINANT_COL

    def __post_init__(self) -> None:
        if self.label_columns is None:
            self.label_columns = LABEL_COLUMNS.copy()
        if self.raw_label_columns is None:
            self.raw_label_columns = RAW_LABEL_COLUMNS.copy()
        if self.optional_metadata_columns is None:
            self.optional_metadata_columns = OPTIONAL_METADATA_COLUMNS.copy()
        total_fraction = self.train_fraction + self.valid_fraction + self.test_fraction
        if not np.isclose(total_fraction, 1.0, atol=1e-6):
            raise ValueError(f"train_fraction + valid_fraction + test_fraction must sum to 1.0; got {total_fraction:.6f}")


class StateManager:
    def __init__(self, state_path: Path):
        self.state_path = state_path
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if self.state_path.exists():
            with open(self.state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"project_initialized": False, "steps_completed": [], "artifacts": {}, "notes": [], "last_updated": None}

    def save(self) -> None:
        self.state["last_updated"] = datetime.utcnow().isoformat()
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    def is_step_done(self, step_name: str) -> bool:
        return step_name in self.state.get("steps_completed", [])

    def mark_step_done(self, step_name: str, artifacts: Optional[Dict[str, str]] = None) -> None:
        if step_name not in self.state["steps_completed"]:
            self.state["steps_completed"].append(step_name)
        if artifacts:
            self.state["artifacts"].update(artifacts)
        self.save()

    def add_note(self, note: str) -> None:
        self.state.setdefault("notes", []).append({"time": datetime.utcnow().isoformat(), "note": note})
        self.save()


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger(STEP_NAME)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    formatter = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def save_json(data: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_df(df: pd.DataFrame, csv_path: Path, parquet_path: Optional[Path], logger: logging.Logger, save_parquet: bool = True) -> None:
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info("Saved CSV: %s", csv_path)
    if parquet_path is not None and save_parquet:
        try:
            df.to_parquet(parquet_path, index=False)
            logger.info("Saved parquet: %s", parquet_path)
        except Exception as e:
            logger.warning("Could not save parquet %s: %s", parquet_path.name, e)


def detect_runtime_environment() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "gpu_available_torch": False,
        "gpu_device_count": 0,
        "gpu_name": None,
        "cuda_version": None,
    }
    try:
        import torch
        info["gpu_available_torch"] = bool(torch.cuda.is_available())
        info["gpu_device_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
    except Exception as e:
        info["torch_probe_error"] = str(e)
    return info


def normalize_text(value: Any) -> str:
    s = str(value).strip().lower().replace("\\", "/")
    s = "_".join(s.split())
    out = []
    for ch in s:
        out.append(ch if ch.isalnum() or ch == "_" else "_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def is_numeric_series(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def safe_mode(series: pd.Series) -> Any:
    vals = series.dropna()
    if vals.empty:
        return None
    modes = vals.mode(dropna=True)
    return modes.iloc[0] if not modes.empty else vals.iloc[0]


def coefficient_of_variation(values: np.ndarray) -> float:
    # FIX I-27: Return std as fallback when mean ~ 0 to avoid silent NaN propagation
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=0)
    if abs(mean_val) <= EPS:
        # Near-zero mean: use std directly as a dispersion measure
        return float(std_val) if std_val > 0 else 0.0
    return float(std_val / abs(mean_val))


def shannon_entropy_from_counts(counts: Sequence[float]) -> float:
    arr = np.asarray(counts, dtype=float)
    arr = arr[arr > 0]
    if arr.size == 0:
        return 0.0
    p = arr / arr.sum()
    return float(-(p * np.log2(p + EPS)).sum())


def iqr(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.percentile(values, 75) - np.percentile(values, 25))


def mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    med = np.median(values)
    return float(np.median(np.abs(values - med)))


def parse_field_strength_to_tesla(value: Any) -> Optional[float]:
    if pd.isna(value):
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    for token in ["tesla", "field", "strength", "id", ":"]:
        s = s.replace(token, " ")
    s = s.replace(",", ".")
    nums = []
    for part in s.split():
        try:
            nums.append(float(part.replace("t", "")))
        except Exception:
            continue
    if not nums:
        return None
    val = nums[0]
    if 1.4 <= val <= 1.6:
        return 1.5
    if 2.8 <= val <= 3.2:
        return 3.0
    return float(val)


def field_strength_to_label(value: Any) -> Optional[str]:
    tesla = parse_field_strength_to_tesla(value)
    if tesla is None:
        return None
    if math.isclose(tesla, 1.5, abs_tol=0.11):
        return "1.5T"
    if math.isclose(tesla, 3.0, abs_tol=0.21):
        return "3T"
    return f"{tesla:g}T"


def volume_entropy_binned(volumes, n_bins: int = 10) -> float:
    # FIX I-07: Entropy of volume distribution using histogram bins (not raw values as counts)
    arr = np.asarray(volumes, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size < 2:
        return 0.0
    counts, _ = np.histogram(arr, bins=n_bins)
    return shannon_entropy_from_counts(counts)


def safe_entropy_from_series(series: pd.Series) -> float:
    vals = series.dropna().astype(str)
    if vals.empty:
        return 0.0
    return shannon_entropy_from_counts(vals.value_counts(dropna=False).to_numpy(dtype=float))


def cliffs_delta_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 2000, seed: int = 42
                    ) -> Tuple[float, float, float]:
    """Cliff's delta between two samples with a percentile bootstrap 95% interval.

    delta = P(X > Y) - P(X < Y), the rank effect size that matches the Mann-Whitney U test:
    0 means the two distributions overlap completely, +/-1 that they are disjoint. Reported
    beside the P value because with 11 patients at 3 T a non-significant test says little
    about the size of any difference.
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return float("nan"), float("nan"), float("nan")

    def delta(a, b):
        # rank-based, so O(n log n) rather than the O(n*m) pairwise form
        ranks = pd.Series(np.concatenate([a, b])).rank().to_numpy()
        ra = ranks[:a.size].sum()
        u = ra - a.size * (a.size + 1) / 2.0          # Mann-Whitney U for a
        return 2.0 * u / (a.size * b.size) - 1.0

    d = delta(x, y)
    rng = np.random.default_rng(seed)
    boots = [delta(rng.choice(x, x.size, replace=True), rng.choice(y, y.size, replace=True))
             for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d), float(lo), float(hi)


def mann_whitney_u_pvalue(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    x_arr = pd.to_numeric(pd.Series(list(x)), errors="coerce").dropna().to_numpy(dtype=float)
    y_arr = pd.to_numeric(pd.Series(list(y)), errors="coerce").dropna().to_numpy(dtype=float)
    if x_arr.size < 3 or y_arr.size < 3:
        return None
    try:
        from scipy.stats import mannwhitneyu
        return float(mannwhitneyu(x_arr, y_arr, alternative="two-sided").pvalue)
    except Exception:
        return None


def chi2_pvalue(table: pd.DataFrame) -> Optional[float]:
    arr = table.to_numpy(dtype=float)
    if arr.size == 0 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return None
    try:
        from scipy.stats import chi2_contingency
        return float(chi2_contingency(arr)[1])
    except Exception:
        return None


def resolve_input_table(cfg: Step04Config) -> Tuple[Path, str]:
    lesion_path = Path(cfg.input_lesion_only_csv)
    full_path = Path(cfg.input_full_csv)
    if lesion_path.exists():
        return lesion_path, "lesion_only"
    if full_path.exists():
        return full_path, "full"
    raise FileNotFoundError(f"Could not find Step 03 output. Expected one of:\n- {lesion_path}\n- {full_path}")


def validate_required_columns(df: pd.DataFrame, required_cols: Sequence[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_identifier_integrity(df: pd.DataFrame, patient_id_col: str, case_id_col: str,
                                   max_missing_fraction: float = 0.02) -> None:
    # FIX I-13: Tolerate small missingness; raise only when > max_missing_fraction
    n_rows = max(len(df), 1)
    for col, label in [(patient_id_col, "patient"), (case_id_col, "case")]:
        if col not in df.columns:
            raise ValueError(f"Required identifier column '{col}' not found in dataset.")
        n_missing = int(df[col].isna().sum())
        frac = n_missing / n_rows
        if frac > max_missing_fraction:
            raise ValueError(
                f"{col} contains {n_missing} missing values ({frac:.1%}). "
                f"Step 04 requires resolvable {label} identifiers."
            )


def infer_radiomics_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for col in df.columns:
        col_norm = normalize_text(col) + "_"
        if is_numeric_series(df[col]) and any(col_norm.startswith(pfx) for pfx in RADIOMICS_PREFIXES):
            cols.append(col)
    return cols


def harmonize_metadata_alias_columns(df: pd.DataFrame, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    alias_to_canonical = {
        "Pixel_Spacing": "Pixel Spacing",
        "Magnetic_Field_Strength_ID": "Magnetic Field Strength ID",
        "Sex": "Gender",
    }
    df = df.copy()
    for alias_col, canonical_col in alias_to_canonical.items():
        if canonical_col not in df.columns and alias_col in df.columns:
            df[canonical_col] = df[alias_col]
            if logger is not None:
                logger.info("Harmonized metadata alias column '%s' -> '%s'.", alias_col, canonical_col)
    return df


def add_scanner_canonical_columns(df: pd.DataFrame, cfg: Step04Config, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    df = df.copy()
    source_col = "Magnetic Field Strength ID" if "Magnetic Field Strength ID" in df.columns else ("Magnetic_Field_Strength_ID" if "Magnetic_Field_Strength_ID" in df.columns else None)
    if source_col is not None:
        df[cfg.field_strength_numeric_col] = df[source_col].apply(parse_field_strength_to_tesla)
        df[cfg.field_strength_label_col] = df[source_col].apply(field_strength_to_label)
        if logger is not None:
            logger.info("Created scanner canonical columns from '%s'.", source_col)
    else:
        df[cfg.field_strength_numeric_col] = np.nan
        df[cfg.field_strength_label_col] = pd.Series([None] * len(df), dtype="object")
        if logger is not None:
            logger.warning("No magnetic field strength column detected; scanner-aware features will be mostly missing.")
    if "Manufacturer" not in df.columns:
        df["Manufacturer"] = pd.Series([None] * len(df), dtype="object")
    return df


def build_label_integrity_report(df: pd.DataFrame, label_cols: Sequence[str]) -> Dict[str, Any]:
    out = {}
    for col in label_cols:
        if col not in df.columns:
            out[col] = {"present": False}
            continue
        vals = df[col].dropna().tolist()
        uniq = sorted(pd.Series(vals).dropna().unique().tolist()) if len(vals) else []
        out[col] = {"present": True, "non_missing": int(df[col].notna().sum()), "unique_values": uniq}
    return out


def build_patient_label_conflict_report(df: pd.DataFrame, cfg: Step04Config) -> Tuple[pd.DataFrame, List[Any]]:
    rows = []
    conflict_patients = []
    for patient_id, g in df.groupby(cfg.patient_id_col, dropna=False):
        row = {cfg.patient_id_col: patient_id}
        has_conflict = False
        for label_col in cfg.label_columns:
            if label_col not in g.columns:
                row[f"{label_col}_status"] = "missing_column"
                row[f"{label_col}_unique_values"] = ""
                continue
            vals = sorted(pd.Series(g[label_col].dropna().astype(float)).unique().tolist())
            row[f"{label_col}_n_non_missing"] = int(g[label_col].notna().sum())
            row[f"{label_col}_unique_values"] = vals
            if len(vals) == 0:
                row[f"{label_col}_status"] = "all_missing"
            elif len(vals) == 1:
                row[f"{label_col}_status"] = "consistent"
            else:
                row[f"{label_col}_status"] = "conflict"
                has_conflict = True
        row["has_any_label_conflict"] = bool(has_conflict)
        if has_conflict:
            conflict_patients.append(patient_id)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(cfg.patient_id_col, kind="stable").reset_index(drop=True), conflict_patients


def build_case_level_table(df: pd.DataFrame, radiomics_cols: Sequence[str], cfg: Step04Config) -> pd.DataFrame:
    if cfg.case_id_col not in df.columns:
        raise ValueError(f"Missing case column: {cfg.case_id_col}")
    case_rows = []
    for case_id, g in df.groupby([cfg.case_id_col], dropna=False):
        row = {
            cfg.case_id_col: case_id,
            cfg.patient_id_col: safe_mode(g[cfg.patient_id_col]) if cfg.patient_id_col in g.columns else None,
            cfg.visit_id_col: safe_mode(g[cfg.visit_id_col]) if cfg.visit_id_col in g.columns else None,
            "n_lesions_in_case": int(g.shape[0]),
        }
        if MASK_CATEGORY_COL in g.columns:
            counts = g[MASK_CATEGORY_COL].astype(str).value_counts(dropna=False)
            for cat, cnt in counts.to_dict().items():
                row[f"case_count_cat_{normalize_text(cat)}"] = int(cnt)
        if cfg.field_strength_numeric_col in g.columns:
            tesla_vals = pd.to_numeric(g[cfg.field_strength_numeric_col], errors="coerce").dropna().values
            row[cfg.field_strength_dominant_col] = float(pd.Series(tesla_vals).mode().iloc[0]) if tesla_vals.size else np.nan
            row["case_n_unique_field_strengths"] = int(pd.Series(tesla_vals).nunique()) if tesla_vals.size else 0
        if cfg.field_strength_label_col in g.columns:
            row[cfg.field_strength_label_col] = safe_mode(g[cfg.field_strength_label_col])
        if "Manufacturer" in g.columns:
            row[cfg.manufacturer_dominant_col] = safe_mode(g["Manufacturer"])
        if "mask_volume_cc" in g.columns:
            vols = pd.to_numeric(g["mask_volume_cc"], errors="coerce").dropna().values
            if vols.size > 0:
                row.update({
                    "case_total_volume_cc": float(np.sum(vols)),
                    "case_mean_volume_cc": float(np.mean(vols)),
                    "case_max_volume_cc": float(np.max(vols)),
                    "case_min_volume_cc": float(np.min(vols)),
                    "case_std_volume_cc": float(np.std(vols, ddof=0)),
                    "case_cv_volume_cc": coefficient_of_variation(vols),
                    # FIX I-07 (case-level): histogram-bin the volumes before entropy;
                    # passing raw continuous volumes as if they were counts is wrong and
                    # was already fixed for the patient-level table (volume_entropy_patient).
                    "case_volume_entropy": volume_entropy_binned(vols),
                })
        for label_col in cfg.label_columns:
            if label_col in g.columns:
                vals = sorted(pd.Series(g[label_col].dropna().astype(float)).unique().tolist())
                row[label_col] = vals[0] if len(vals) == 1 else np.nan
        for raw_label_col in cfg.raw_label_columns:
            if raw_label_col in g.columns:
                row[raw_label_col] = safe_mode(g[raw_label_col])
        for meta_col in cfg.optional_metadata_columns:
            if meta_col in g.columns and meta_col not in row:
                if is_numeric_series(g[meta_col]):
                    vals = pd.to_numeric(g[meta_col], errors="coerce").dropna().values
                    row[f"case_meta_{normalize_text(meta_col)}_mean"] = float(np.mean(vals)) if vals.size else np.nan
                else:
                    row[f"case_meta_{normalize_text(meta_col)}_mode"] = safe_mode(g[meta_col])
        if radiomics_cols:
            rad_df = g[list(radiomics_cols)].apply(pd.to_numeric, errors="coerce")
            for col in radiomics_cols:
                vals = rad_df[col].dropna().values
                if vals.size == 0:
                    continue
                prefix = f"case_{normalize_text(col)}"
                row[f"{prefix}_mean"] = float(np.mean(vals))
                row[f"{prefix}_std"] = float(np.std(vals, ddof=0))
                row[f"{prefix}_min"] = float(np.min(vals))
                row[f"{prefix}_max"] = float(np.max(vals))
                row[f"{prefix}_cv"] = coefficient_of_variation(vals)
        case_rows.append(row)
    return pd.DataFrame(case_rows)


def compute_pairwise_radiomics_metrics(rad_matrix: np.ndarray) -> Dict[str, float]:
    nan_result = {"pairwise_euclidean_mean": np.nan, "pairwise_euclidean_max": np.nan, "pairwise_euclidean_std": np.nan}
    if rad_matrix.ndim != 2 or rad_matrix.shape[0] < 2 or rad_matrix.shape[1] == 0:
        return nan_result
    mat = rad_matrix.astype(float).copy()
    col_means = np.where(np.isfinite(mat).any(axis=0), np.nanmean(np.where(np.isfinite(mat), mat, np.nan), axis=0), 0.0)
    bad = ~np.isfinite(mat)
    mat[bad] = np.take(col_means, np.where(bad)[1])
    col_means_imp = np.mean(mat, axis=0)
    col_stds = np.std(mat, axis=0, ddof=0)
    col_stds[col_stds <= EPS] = 1.0
    mat = (mat - col_means_imp) / col_stds
    dists = [float(np.linalg.norm(mat[i] - mat[j])) for i in range(mat.shape[0]) for j in range(i + 1, mat.shape[0])]
    if not dists:
        return nan_result
    arr = np.asarray(dists, dtype=float)
    return {"pairwise_euclidean_mean": float(np.mean(arr)), "pairwise_euclidean_max": float(np.max(arr)), "pairwise_euclidean_std": float(np.std(arr, ddof=0))}


def add_dominant_lesion_features(group: pd.DataFrame, row: Dict[str, Any]) -> None:
    if "mask_volume_cc" not in group.columns:
        return
    g = group.copy()
    g["mask_volume_cc"] = pd.to_numeric(g["mask_volume_cc"], errors="coerce")
    row["n_lesions_with_valid_volume"] = int(g["mask_volume_cc"].notna().sum())
    g = g.dropna(subset=["mask_volume_cc"]).sort_values("mask_volume_cc", ascending=False)
    if g.empty:
        return
    largest = float(g.iloc[0]["mask_volume_cc"])
    row["largest_lesion_volume_cc"] = largest
    if g.shape[0] >= 2:
        second = float(g.iloc[1]["mask_volume_cc"])
        row["second_largest_lesion_volume_cc"] = second
        row["largest_to_second_largest_volume_ratio"] = float(largest / second) if second > 0 else np.nan
    else:
        row["second_largest_lesion_volume_cc"] = np.nan
        row["largest_to_second_largest_volume_ratio"] = np.nan
    total = float(g["mask_volume_cc"].sum())
    row["largest_lesion_burden_fraction"] = float(largest / total) if total > 0 else np.nan


def build_patient_level_table(df: pd.DataFrame, radiomics_cols: Sequence[str], cfg: Step04Config) -> pd.DataFrame:
    patient_rows = []
    # Fixed, cohort-level feature order so that every patient contributes the same columns.
    cohort_pairwise_cols: List[str] = []
    if radiomics_cols:
        _rad_all = df[list(radiomics_cols)].apply(pd.to_numeric, errors="coerce")
        cohort_pairwise_cols = (_rad_all.var(axis=0, ddof=0).fillna(0.0)
                                .sort_values(ascending=False)
                                .head(cfg.max_pairwise_features).index.tolist())
    for patient_id, g in df.groupby(cfg.patient_id_col, dropna=False):
        row = {
            cfg.patient_id_col: patient_id,
            "n_rows_patient": int(g.shape[0]),
            "n_cases_patient": int(g[cfg.case_id_col].nunique(dropna=True)) if cfg.case_id_col in g.columns else np.nan,
            "n_visits_patient": int(g[cfg.visit_id_col].nunique(dropna=True)) if cfg.visit_id_col in g.columns else np.nan,
        }
        if cfg.segmentation_col in g.columns:
            row["n_unique_segmentation_names_patient"] = int(g[cfg.segmentation_col].nunique(dropna=True))
        if MASK_CATEGORY_COL in g.columns:
            cat_counts = g[MASK_CATEGORY_COL].astype(str).value_counts(dropna=False)
            for cat, cnt in cat_counts.to_dict().items():
                row[f"count_cat_{normalize_text(cat)}"] = int(cnt)
            row["mask_category_entropy_patient"] = shannon_entropy_from_counts(cat_counts.to_numpy())
        for label_col in cfg.label_columns:
            if label_col in g.columns:
                vals = sorted(pd.Series(g[label_col].dropna().astype(float)).unique().tolist())
                row[label_col] = vals[0] if len(vals) == 1 else np.nan
        for raw_label_col in cfg.raw_label_columns:
            if raw_label_col in g.columns:
                row[raw_label_col] = safe_mode(g[raw_label_col])

        if cfg.field_strength_numeric_col in g.columns:
            tesla_vals = pd.to_numeric(g[cfg.field_strength_numeric_col], errors="coerce").dropna().values
            row[cfg.field_strength_dominant_col] = float(pd.Series(tesla_vals).mode().iloc[0]) if tesla_vals.size else np.nan
            row["mean_field_strength_t"] = float(np.mean(tesla_vals)) if tesla_vals.size else np.nan
            row["std_field_strength_t"] = float(np.std(tesla_vals, ddof=0)) if tesla_vals.size else np.nan
            row["min_field_strength_t"] = float(np.min(tesla_vals)) if tesla_vals.size else np.nan
            row["max_field_strength_t"] = float(np.max(tesla_vals)) if tesla_vals.size else np.nan
            row["n_unique_field_strength_patient"] = int(pd.Series(tesla_vals).nunique()) if tesla_vals.size else 0
        if cfg.field_strength_label_col in g.columns:
            row[cfg.field_strength_label_col] = safe_mode(g[cfg.field_strength_label_col])
            row["entropy_magnetic_field_strength_patient"] = safe_entropy_from_series(g[cfg.field_strength_label_col])
        if "Manufacturer" in g.columns:
            row[cfg.manufacturer_dominant_col] = safe_mode(g["Manufacturer"])
            row["n_unique_manufacturer_patient"] = int(g["Manufacturer"].nunique(dropna=True))
            row["entropy_manufacturer_patient"] = safe_entropy_from_series(g["Manufacturer"])

        for meta_col in cfg.optional_metadata_columns:
            if meta_col not in g.columns:
                continue
            if is_numeric_series(g[meta_col]):
                vals = pd.to_numeric(g[meta_col], errors="coerce").dropna().values
                prefix = f"patient_meta_{normalize_text(meta_col)}"
                if vals.size > 0:
                    row[f"{prefix}_mean"] = float(np.mean(vals))
                    row[f"{prefix}_std"] = float(np.std(vals, ddof=0))
                    row[f"{prefix}_min"] = float(np.min(vals))
                    row[f"{prefix}_max"] = float(np.max(vals))
                else:
                    row[f"{prefix}_mean"] = np.nan
            else:
                row[f"patient_meta_{normalize_text(meta_col)}_mode"] = safe_mode(g[meta_col])
                row[f"patient_meta_{normalize_text(meta_col)}_n_unique"] = int(g[meta_col].nunique(dropna=True))

        if "mask_volume_cc" in g.columns:
            volumes = pd.to_numeric(g["mask_volume_cc"], errors="coerce").dropna().values
            if volumes.size > 0:
                row.update({
                    "total_tumor_volume_cc": float(np.sum(volumes)),
                    "mean_lesion_volume_cc": float(np.mean(volumes)),
                    "median_lesion_volume_cc": float(np.median(volumes)),
                    "std_lesion_volume_cc": float(np.std(volumes, ddof=0)),
                    "min_lesion_volume_cc": float(np.min(volumes)),
                    "max_lesion_volume_cc": float(np.max(volumes)),
                    "cv_lesion_volume_cc": coefficient_of_variation(volumes),
                    "iqr_lesion_volume_cc": iqr(volumes),
                    "mad_lesion_volume_cc": mad(volumes),
                    "volume_entropy_patient": volume_entropy_binned(volumes),  # FIX I-07: use histogram bins
                    "n_small_lesions_le_0_1cc": int(np.sum(volumes <= 0.1)),
                    "n_medium_lesions_gt_0_1cc_le_1cc": int(np.sum((volumes > 0.1) & (volumes <= 1.0))),
                    "n_large_lesions_gt_1cc": int(np.sum(volumes > 1.0)),
                })
        add_dominant_lesion_features(g, row)

        for col in ["mask_descriptor_canonical", cfg.segmentation_col, "Manufacturer", "Pixel Spacing", cfg.field_strength_label_col]:
            if col in g.columns:
                counts = g[col].astype(str).fillna("missing").value_counts(dropna=False)
                row[f"n_unique_{normalize_text(col)}_patient"] = int(g[col].nunique(dropna=True))
                row[f"entropy_{normalize_text(col)}_patient"] = shannon_entropy_from_counts(counts.to_numpy())

        if radiomics_cols:
            rad_df_full = g[list(radiomics_cols)].apply(pd.to_numeric, errors="coerce")
            within_var = rad_df_full.var(axis=0, ddof=0).fillna(0.0)
            if cfg.fixed_patient_feature_set:
                top_var_cols = list(radiomics_cols)
            else:
                top_var_cols = within_var.sort_values(ascending=False).head(cfg.top_patient_rad_features).index.tolist()
            largest_idx = None
            if cfg.radiomic_aggregation == "largest":
                vol = pd.to_numeric(g["mask_volume_cc"], errors="coerce") if "mask_volume_cc" in g.columns else None
                if vol is None or vol.notna().sum() == 0:
                    raise RuntimeError(f"Patient {patient_id}: no lesion volume, cannot pick the largest lesion.")
                largest_idx = vol.idxmax()   # first occurrence on ties; the input order is fixed
            for col in top_var_cols:
                vals = rad_df_full[col].dropna().values
                if vals.size == 0:
                    continue
                prefix = f"patient_{normalize_text(col)}"
                if cfg.radiomic_aggregation == "median":
                    row[f"{prefix}_median"] = float(np.median(vals))
                    continue
                if cfg.radiomic_aggregation == "largest":
                    v = rad_df_full.at[largest_idx, col]
                    row[f"{prefix}_largest"] = float(v) if pd.notna(v) else np.nan
                    continue
                row[f"{prefix}_mean"] = float(np.mean(vals))
                row[f"{prefix}_std"] = float(np.std(vals, ddof=0))
                row[f"{prefix}_min"] = float(np.min(vals))
                row[f"{prefix}_max"] = float(np.max(vals))
                row[f"{prefix}_cv"] = coefficient_of_variation(vals)
                row[f"{prefix}_iqr"] = iqr(vals)
            pairwise_cols = (cohort_pairwise_cols if cfg.fixed_patient_feature_set
                             else within_var.sort_values(ascending=False).head(cfg.max_pairwise_features).index.tolist())
            rad_df_pair = rad_df_full[pairwise_cols].dropna(axis=0, how="all")
            if not rad_df_pair.empty:
                row.update(compute_pairwise_radiomics_metrics(rad_df_pair.values))
            cv_values, std_values = [], []
            for col in radiomics_cols:
                vals = rad_df_full[col].dropna().values
                if vals.size < 2:
                    continue
                cv_val = coefficient_of_variation(vals)
                std_val = float(np.std(vals, ddof=0))
                if np.isfinite(cv_val):
                    cv_values.append(cv_val)
                if np.isfinite(std_val):
                    std_values.append(std_val)
            row["global_radiomics_cv_mean"] = float(np.mean(cv_values)) if cv_values else np.nan
            row["global_radiomics_cv_max"] = float(np.max(cv_values)) if cv_values else np.nan
            row["global_radiomics_std_mean"] = float(np.mean(std_values)) if std_values else np.nan
        patient_rows.append(row)

    patient_df = pd.DataFrame(patient_rows)
    for label_col in cfg.label_columns:
        if label_col in patient_df.columns:
            patient_df[f"target_{label_col.replace('_bin', '').lower()}"] = patient_df[label_col]
    return patient_df


def build_stratification_key(patient_df: pd.DataFrame, logger: Optional[logging.Logger] = None, include_field_strength: bool = True, field_strength_col: str = FIELD_STRENGTH_DOMINANT_COL) -> pd.Series:
    target_cols = ["target_er", "target_pr", "target_her2"]
    for col in target_cols:
        if col not in patient_df.columns and logger:
            logger.warning("Target column '%s' absent from patient_df — all patients will encode as 'M' for this target in the stratification key.", col)
        elif col in patient_df.columns and patient_df[col].notna().sum() == 0 and logger:
            logger.warning("Target column '%s' is present but entirely missing — stratification key will encode all patients as 'M' for this target.", col)

    def key_from_row(row: pd.Series) -> str:
        parts = []
        for col in target_cols:
            if col not in row.index or pd.isna(row[col]):
                parts.append("M")
            else:
                parts.append(str(int(row[col])))
        if include_field_strength:
            if field_strength_col not in row.index or pd.isna(row[field_strength_col]):
                parts.append("FSM")
            else:
                parts.append(f"FS{float(row[field_strength_col]):.1f}")
        return "_".join(parts)

    return patient_df.apply(key_from_row, axis=1)


def split_patients_stratified(patient_df: pd.DataFrame, patient_id_col: str, seed: int, train_fraction: float, valid_fraction: float, test_fraction: float, min_patients_per_stratum: int, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    total_fraction = train_fraction + valid_fraction + test_fraction
    if not np.isclose(total_fraction, 1.0, atol=1e-6):
        raise ValueError(f"Split fractions must sum to 1.0, got {total_fraction:.6f}")
    work = patient_df[[patient_id_col]].copy()
    work["stratum_key"] = patient_df["stratum_key"] if "stratum_key" in patient_df.columns else build_stratification_key(patient_df, logger)
    rng = random.Random(seed)
    train_ids, valid_ids, test_ids, collapsed_strata = [], [], [], []
    for stratum, g in work.groupby("stratum_key", dropna=False):
        ids = sorted(g[patient_id_col].dropna().drop_duplicates().tolist(), key=lambda x: str(x))
        rng.shuffle(ids)
        n = len(ids)
        if n < min_patients_per_stratum:
            train_ids.extend(ids)
            collapsed_strata.append(f"{stratum}(n={n})")
            continue
        n_train = max(1, int(round(n * train_fraction)))
        n_valid = int(round(n * valid_fraction))
        if n_valid < 1 and n >= 3:
            n_valid = 1
        if n_train + n_valid >= n:
            n_valid = max(1, n - n_train - 1)
            if n_train + n_valid >= n and n >= 2:
                n_train = max(1, n - n_valid - 1)
        train_ids.extend(ids[:n_train])
        valid_ids.extend(ids[n_train:n_train + n_valid])
        test_ids.extend(ids[n_train + n_valid:])
    if collapsed_strata and logger:
        logger.warning(
            "The following strata had fewer than min_patients_per_stratum=%d patients "
            "and were collapsed entirely into train: %s",
            min_patients_per_stratum, collapsed_strata
        )
        # FIX I-08: Warn if collapsed fraction is large — may bias class balance
        try:
            n_collapsed_patients = sum(
                int(s.split("n=")[1].rstrip(")")) for s in collapsed_strata if "n=" in s
            )
            total_patients = len(patient_df)
            frac = n_collapsed_patients / max(total_patients, 1)
            if frac > 0.20:
                logger.warning(
                    "WARNING: %.0f%% of patients (%d/%d) are in collapsed strata and assigned "
                    "entirely to train. Validation/test results may be unreliable for rare subgroups.",
                    frac * 100, n_collapsed_patients, total_patients
                )
        except Exception:
            pass
    train_ids_set, valid_ids_set, test_ids_set = set(train_ids), set(valid_ids), set(test_ids)
    overlap = (train_ids_set & valid_ids_set) | (train_ids_set & test_ids_set) | (valid_ids_set & test_ids_set)
    if overlap:
        raise RuntimeError(f"Overlap detected across splits for patient IDs: {list(overlap)[:10]}")
    split_df = patient_df[[patient_id_col]].copy()
    def assign_split(pid: Any) -> str:
        if pd.isna(pid):
            return "unassigned"
        if pid in train_ids_set:
            return "train"
        if pid in valid_ids_set:
            return "valid"
        if pid in test_ids_set:
            return "test"
        return "unassigned"
    split_df["split"] = split_df[patient_id_col].apply(assign_split)
    return split_df.sort_values(patient_id_col, kind="stable").reset_index(drop=True)


def build_feature_dictionary(patient_df: pd.DataFrame, cfg: Step04Config) -> Dict[str, Any]:
    id_cols = {cfg.patient_id_col, "split", "stratum_key"}
    target_cols = [f"target_{c.replace('_bin', '').lower()}" for c in cfg.label_columns]
    label_cols = set(cfg.label_columns + cfg.raw_label_columns + target_cols)
    numeric_features, categorical_features, high_missingness_excluded = [], [], []
    for col in patient_df.columns:
        if col in id_cols or col in label_cols:
            continue
        if is_numeric_series(patient_df[col]):
            miss_ratio = float(patient_df[col].isna().mean())
            if miss_ratio >= cfg.high_missingness_threshold:
                high_missingness_excluded.append(col)
            else:
                numeric_features.append(col)
        else:
            categorical_features.append(col)
    scanner_covariate_columns = [c for c in SCANNER_COVARIATE_PRIORITY if c in patient_df.columns]
    field_strength_only_candidates = [c for c in scanner_covariate_columns if "field_strength" in c or c == cfg.field_strength_label_col]
    numeric_features_no_scanner = [c for c in numeric_features if c not in scanner_covariate_columns]

    # Explicit feature blocks. Dropping the eight named scanner covariates is NOT enough to make a
    # set "radiomics only": the patient-level table also carries field strength under other names
    # (mean/min/max/std_field_strength_t, patient_meta_magnetic_field_strength_id_*), pixel spacing
    # and slice counts. Those are acquisition descriptors and belong with the scanner block, or the
    # radiomics arm silently competes with its own control. Blocks are defined by construction:
    # radiomic = the per-lesion IBSI features aggregated to the patient; acquisition = anything
    # naming a scanner, field strength, pixel spacing, slice geometry or resolution provenance;
    # burden/study = lesion counts, volumes and case bookkeeping.
    ACQ_TOKENS = ("field_strength", "manufacturer", "pixel_spacing", "scanner", "slice",
                  "thickness", "resolution_confidence", "resolution_source")
    # Quantities step_03 derives from the segmentation mask. "slice" is in ACQ_TOKENS to catch
    # slice thickness and geometry, but it also matches n_slices_involved, which counts the
    # slices a lesion spans -- lesion extent, not an acquisition parameter. The acquisition
    # block is the control against the radiomic block, so a column describing the lesion must
    # not sit inside it. Excluded by name rather than by dropping the token, so that a real
    # slice_thickness column would still be classified as acquisition.
    MASK_DERIVED_TOKENS = ("n_slices_involved", "mask_voxel_count", "mask_volume", "bbox_",
                           "n_components", "elongation_ratio", "mask_fill_ratio")
    is_mask_derived = lambda c: any(t in c for t in MASK_DERIVED_TOKENS)
    radiomic_block = [c for c in numeric_features if c.startswith("patient_original_")]
    acquisition_block = sorted(
        {c for c in numeric_features
         if c not in radiomic_block and any(t in c for t in ACQ_TOKENS) and not is_mask_derived(c)}
        | {c for c in set(scanner_covariate_columns) & set(numeric_features)
           if not is_mask_derived(c)})
    burden_study_block = [c for c in numeric_features
                          if c not in radiomic_block and c not in acquisition_block]
    return {
        "patient_id_column": cfg.patient_id_col,
        "numeric_feature_columns": sorted(numeric_features),
        "numeric_feature_columns_no_scanner_covariates": sorted(numeric_features_no_scanner),
        "radiomic_block_columns": sorted(radiomic_block),
        "acquisition_block_columns": sorted(acquisition_block),
        "burden_study_block_columns": sorted(burden_study_block),
        "categorical_feature_columns": sorted(categorical_features),
        "scanner_covariate_columns": sorted(scanner_covariate_columns),
        "field_strength_only_candidate_columns": sorted(field_strength_only_candidates),
        "target_columns": [c for c in target_cols if c in patient_df.columns],
        "label_columns": [c for c in cfg.label_columns if c in patient_df.columns],
        "raw_label_columns": [c for c in cfg.raw_label_columns if c in patient_df.columns],
        "high_missingness_excluded_columns": sorted(high_missingness_excluded),
        "high_missingness_threshold_used": cfg.high_missingness_threshold,
    }


def build_missingness_report(patient_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in patient_df.columns:
        rows.append({
            "column_name": col,
            "dtype": str(patient_df[col].dtype),
            "n_missing": int(patient_df[col].isna().sum()),
            "missing_ratio": float(patient_df[col].isna().mean()),
            "n_unique_non_null": int(patient_df[col].dropna().nunique()),
        })
    return pd.DataFrame(rows).sort_values(["missing_ratio", "column_name"], ascending=[False, True])


def build_scanner_confounding_audit(lesion_df: pd.DataFrame, patient_df: pd.DataFrame, radiomics_cols: Sequence[str], cfg: Step04Config) -> Dict[str, Any]:
    audit = {"enabled": True, "field_strength_column_used": cfg.field_strength_numeric_col, "patient_dominant_field_strength_column": cfg.field_strength_dominant_col}
    if cfg.field_strength_numeric_col not in lesion_df.columns:
        audit["available"] = False
        audit["reason"] = "field_strength_column_missing"
        return audit
    lesion_non_null = pd.to_numeric(lesion_df[cfg.field_strength_numeric_col], errors="coerce").notna().sum()
    audit["available"] = bool(lesion_non_null > 0)
    audit["n_lesion_rows_with_field_strength"] = int(lesion_non_null)
    if cfg.field_strength_label_col in lesion_df.columns:
        audit["lesion_field_strength_counts"] = lesion_df[cfg.field_strength_label_col].fillna("missing").value_counts(dropna=False).to_dict()
    if cfg.field_strength_dominant_col in patient_df.columns:
        audit["patient_field_strength_counts"] = patient_df[cfg.field_strength_dominant_col].astype(str).fillna("missing").value_counts(dropna=False).to_dict()
    label_vs_strength = {}
    for target_col in ["target_er", "target_pr", "target_her2"]:
        if target_col not in patient_df.columns or cfg.field_strength_dominant_col not in patient_df.columns:
            continue
        sub = patient_df[[target_col, cfg.field_strength_dominant_col]].dropna()
        if sub.empty:
            continue
        ctab = pd.crosstab(sub[cfg.field_strength_dominant_col], sub[target_col])
        label_vs_strength[target_col] = {
            "contingency_table": {str(idx): {str(k): int(v) for k, v in row.items()} for idx, row in ctab.to_dict(orient="index").items()},
            "p_value_chi2": chi2_pvalue(ctab),
        }
    audit["label_vs_field_strength"] = label_vs_strength

    def _bh(pvals):
        """Benjamini-Hochberg adjusted P values, same order as the input."""
        arr = np.asarray(pvals, dtype=float)
        if arr.size == 0:
            return arr
        order = np.argsort(arr)
        adj = np.empty_like(arr)
        ranks = np.arange(1, arr.size + 1)
        adj[order] = np.minimum.accumulate((arr[order] * arr.size / ranks)[::-1])[::-1]
        return np.clip(adj, 0.0, 1.0)

    def _screen(frame, group_col, feature_cols, id_col=None):
        """Mann-Whitney U per feature between the two field-strength groups."""
        out = []
        if group_col not in frame.columns:
            return out, {}
        labels = frame[group_col].astype(str)
        numeric = pd.to_numeric(frame[group_col], errors="coerce")
        g1 = (numeric == 1.5) | labels.str.startswith("1.5")
        g2 = (numeric == 3.0) | labels.str.startswith("3")
        meta = {"n_group_1_5T": int(g1.sum()), "n_group_3T": int(g2.sum())}
        if id_col and id_col in frame.columns:
            meta["n_patients_1_5T"] = int(frame.loc[g1, id_col].nunique())
            meta["n_patients_3T"] = int(frame.loc[g2, id_col].nunique())
        for col in feature_cols:
            x = pd.to_numeric(frame.loc[g1, col], errors="coerce").dropna()
            y = pd.to_numeric(frame.loc[g2, col], errors="coerce").dropna()
            p = mann_whitney_u_pvalue(x, y)
            if p is None or x.empty or y.empty:
                continue
            delta, lo, hi = cliffs_delta_ci(x.to_numpy(), y.to_numpy(), seed=cfg.global_seed)
            out.append({
                "feature": col,
                "p_value_mannwhitney": float(p),
                "median_group_1_5T": float(np.median(x)),
                "median_group_3T": float(np.median(y)),
                "abs_median_difference": float(abs(np.median(x) - np.median(y))),
                # Effect size beside the P value: with 11 patients at 3 T, "not significant"
                # and "no difference" are different statements, and only the effect size and
                # its interval say which one the data support.
                "cliffs_delta": float(delta),
                "cliffs_delta_ci_lo": float(lo),
                "cliffs_delta_ci_hi": float(hi),
            })
        if out:
            for row, q in zip(out, _bh([r["p_value_mannwhitney"] for r in out])):
                row["p_value_bh"] = float(q)
        return out, meta

    # PRIMARY: patient level. The unit of analysis is the patient, because lesions from one
    # patient share a scanner, a session and a tumour biology, and are not independent.
    patient_feats = [c for c in patient_df.columns
                     if c.startswith("patient_original_") and c.endswith("_mean")]
    rows, pmeta = _screen(patient_df, cfg.field_strength_dominant_col, patient_feats)
    audit["radiomics_shift_screening"] = rows
    audit["radiomics_shift_unit_of_analysis"] = "patient"
    audit["radiomics_shift_test"] = "two-sided Mann-Whitney U, 1.5 T versus 3 T"
    audit["radiomics_shift_group_sizes"] = pmeta
    audit["n_radiomics_shift_features_tested"] = int(len(rows))
    audit["n_radiomics_shift_features_p_lt_0_05"] = int(sum(1 for r in rows if r["p_value_mannwhitney"] < 0.05))
    audit["n_radiomics_shift_features_p_lt_0_0001"] = int(sum(1 for r in rows if r["p_value_mannwhitney"] < 1e-4))
    audit["n_radiomics_shift_features_bh_lt_0_05"] = int(sum(1 for r in rows if r.get("p_value_bh", 1.0) < 0.05))

    # CONTRAST: the same screening on lesions, which treats every lesion as independent.
    # Reported so the inflation is visible rather than hidden -- not as a finding.
    lesion_rows, lmeta = _screen(lesion_df, cfg.field_strength_label_col, radiomics_cols,
                                 id_col="patient_base")
    audit["radiomics_shift_screening_lesion_level_contrast"] = {
        "note": ("Lesions treated as independent, which they are not: this is reported only "
                 "to show how much the apparent effect inflates when the unit of analysis is "
                 "wrong."),
        "group_sizes": lmeta,
        "n_features_tested": int(len(lesion_rows)),
        "n_p_lt_0_05": int(sum(1 for r in lesion_rows if r["p_value_mannwhitney"] < 0.05)),
        "n_p_lt_0_0001": int(sum(1 for r in lesion_rows if r["p_value_mannwhitney"] < 1e-4)),
        "n_bh_lt_0_05": int(sum(1 for r in lesion_rows if r.get("p_value_bh", 1.0) < 0.05)),
    }
    return audit


def build_split_field_strength_balance(patient_df: pd.DataFrame, cfg: Step04Config) -> Dict[str, Any]:
    if cfg.field_strength_dominant_col not in patient_df.columns or "split" not in patient_df.columns:
        return {}
    work = patient_df[["split", cfg.field_strength_dominant_col]].copy()
    work["field_strength_label"] = work[cfg.field_strength_dominant_col].apply(field_strength_to_label)
    table = pd.crosstab(work["split"], work["field_strength_label"].fillna("missing"), dropna=False)
    return {str(idx): {str(k): int(v) for k, v in row.items()} for idx, row in table.to_dict(orient="index").items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 04 feature engineering")
    ap.add_argument("--one-session", action="store_true",
                    help="Sensitivity analysis: keep only the earliest examination per patient and "
                         "write *_onesession outputs without touching the primary tables.")
    ap.add_argument("--force", action="store_true", help="Re-run even if the step is marked complete.")
    ap.add_argument("--reuse-split-from", default=None, metavar="CSV",
                    help="Take the train/valid/test assignment from this patient-level table instead "
                         "of re-deriving it, so a sensitivity run differs only in the intended respect.")
    ap.add_argument("--aggregation", default="six_stat", choices=RADIOMIC_AGGREGATIONS,
                    help="How lesion-level radiomic features are summarised per patient. Anything but "
                         "six_stat is a sensitivity run and writes *_agg_<mode> outputs.")
    args = ap.parse_args()
    if args.one_session and args.aggregation != "six_stat":
        raise SystemExit("--one-session and --aggregation are separate sensitivity analyses; run one at a time.")
    cfg = Step04Config(one_session_per_patient=args.one_session, radiomic_aggregation=args.aggregation)
    suffix = ONE_SESSION_SUFFIX if args.one_session else ""
    if args.aggregation != "six_stat":
        suffix = f"_agg_{args.aggregation}"
    sensitivity_run = bool(suffix)
    set_global_seed(cfg.global_seed)

    processed_dir = PROJECT_ROOT / "data" / "processed"
    metadata_dir = PROJECT_ROOT / "metadata"
    logs_dir = PROJECT_ROOT / "logs"
    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    for p in [processed_dir, metadata_dir, logs_dir, checkpoints_dir]:
        ensure_dir(p)

    log_file = logs_dir / f"{STEP_NAME}.log"
    logger = setup_logger(log_file)
    state = StateManager(checkpoints_dir / "pipeline_state.json")
    logger.info("=" * 80)
    logger.info("Starting %s", STEP_NAME)

    if not state.is_step_done("step_03_segmentation_harmonization"):
        raise RuntimeError("step_03_segmentation_harmonization must be completed before Step 04.")
    if state.is_step_done(STEP_NAME) and not (args.force or sensitivity_run):
        logger.info("%s already completed. Nothing to do.", STEP_NAME)
        logger.info("=" * 80)
        return

    input_path, input_mode = resolve_input_table(cfg)
    logger.info("Using Step 03 input: %s | mode=%s", input_path, input_mode)

    df = pd.read_csv(input_path)
    logger.info("Loaded Step 03 table with shape: %s", df.shape)

    df = harmonize_metadata_alias_columns(df, logger)
    df = add_scanner_canonical_columns(df, cfg, logger)
    validate_required_columns(df, [cfg.patient_id_col, cfg.case_id_col])
    validate_identifier_integrity(df, cfg.patient_id_col, cfg.case_id_col)

    runtime_info = detect_runtime_environment()
    logger.info("Runtime environment: %s", runtime_info)

    if cfg.analysis_flag_col in df.columns and input_mode == "full":
        logger.info("Full Step 03 table detected; filtering to analysis candidates.")
        df = df[df[cfg.analysis_flag_col] == 1].copy()
        logger.info("Filtered primary analysis table shape: %s", df.shape)

    if getattr(cfg, "one_session_per_patient", False):
        before_rows, before_cases = len(df), df[cfg.case_id_col].nunique()
        vi = pd.to_numeric(df[cfg.visit_id_col], errors="coerce")
        first = vi.groupby(df[cfg.patient_id_col]).transform("min")
        df = df[vi.eq(first)].copy()
        logger.info("One-session sensitivity: kept the earliest visit per patient | rows %d -> %d, "
                    "cases %d -> %d, patients %d", before_rows, len(df), before_cases,
                    df[cfg.case_id_col].nunique(), df[cfg.patient_id_col].nunique())

    radiomics_cols = infer_radiomics_columns(df)
    logger.info("Detected radiomics columns: %d", len(radiomics_cols))

    label_report = build_label_integrity_report(df, cfg.label_columns)
    conflict_df, conflict_patients = build_patient_label_conflict_report(df, cfg)
    conflict_report_path = metadata_dir / f"step04_patient_label_conflict_report{suffix}.csv"
    conflict_df.to_csv(conflict_report_path, index=False, encoding="utf-8-sig")
    logger.info("Patients with label conflicts: %d", len(conflict_patients))
    if conflict_patients:
        logger.warning("Dropping %d patients with inconsistent labels: %s", len(conflict_patients), conflict_patients[:10])
        df = df[~df[cfg.patient_id_col].isin(conflict_patients)].copy()
        logger.info("Shape after conflict exclusion: %s", df.shape)

    case_df = build_case_level_table(df, radiomics_cols, cfg).sort_values([cfg.patient_id_col, cfg.case_id_col], kind="stable").reset_index(drop=True)
    logger.info("Built case-level table with shape: %s", case_df.shape)

    patient_df = build_patient_level_table(df, radiomics_cols, cfg).sort_values(cfg.patient_id_col, kind="stable").reset_index(drop=True)
    logger.info("Built patient-level table with shape: %s", patient_df.shape)

    patient_df["stratum_key"] = build_stratification_key(
        patient_df,
        logger=logger,
        include_field_strength=cfg.include_field_strength_in_stratification,
        field_strength_col=cfg.field_strength_dominant_col,
    )

    splits_df = split_patients_stratified(
        patient_df, cfg.patient_id_col, cfg.global_seed,
        cfg.train_fraction, cfg.valid_fraction, cfg.test_fraction,
        cfg.min_patients_per_stratum, logger
    )
    patient_df = patient_df.merge(splits_df, on=cfg.patient_id_col, how="left")
    if args.reuse_split_from:
        # A sensitivity run must differ from the primary analysis in exactly one respect. The
        # stratification uses field strength, which the session filter can change, so re-deriving
        # the split moved 21 of 139 patients; importing the primary split keeps the comparison
        # clean and is why this option exists.
        ref = pd.read_csv(args.reuse_split_from, low_memory=False)[[cfg.patient_id_col, "split"]]
        before = patient_df["split"].copy()
        patient_df = patient_df.drop(columns=["split"]).merge(ref, on=cfg.patient_id_col, how="left")
        if patient_df["split"].isna().any():
            raise RuntimeError("reuse-split-from does not cover every patient in this run.")
        changed = int((before.values != patient_df["split"].values).sum())
        logger.info("Reused the split from %s (%d of %d patients differ from the re-derived split).",
                    Path(args.reuse_split_from).name, changed, len(patient_df))
    if patient_df["split"].isna().any() or (patient_df["split"] == "unassigned").any():
        raise RuntimeError("Unexpected unassigned patient split detected after split generation.")
    logger.info("Assigned stratified patient-level train/valid/test splits.")
    logger.info("Split counts: %s", patient_df["split"].value_counts(dropna=False).to_dict())

    feature_dict = build_feature_dictionary(patient_df, cfg)
    missingness_df = build_missingness_report(patient_df)

    scanner_audit = {"enabled": False}
    scanner_shift_df = pd.DataFrame()
    if cfg.enable_scanner_confounding_audit:
        scanner_audit = build_scanner_confounding_audit(df, patient_df, radiomics_cols, cfg)
        scanner_shift_df = pd.DataFrame(scanner_audit.get("radiomics_shift_screening", []))

    step04_config_path = metadata_dir / f"step04_config{suffix}.json"
    runtime_path = metadata_dir / f"step04_runtime_environment{suffix}.json"
    label_report_path = metadata_dir / f"step04_label_integrity_report{suffix}.json"
    feature_dict_path = metadata_dir / f"step04_feature_dictionary{suffix}.json"
    # Suffixed like every other step-04 output. Unsuffixed, a --one-session run overwrote the
    # pooled audit the manuscript reports, replacing 720 lesions from 12 patients at 3 T with
    # 22 lesions from 9, and nothing downstream could tell.
    scanner_audit_json_path = metadata_dir / f"step04_scanner_confounding_audit{suffix}.json"
    scanner_shift_csv_path = metadata_dir / f"step04_scanner_radiomics_shift_screening{suffix}.csv"

    save_json(asdict(cfg), step04_config_path)
    save_json(runtime_info, runtime_path)
    save_json(label_report, label_report_path)
    save_json(feature_dict, feature_dict_path)
    save_json(scanner_audit, scanner_audit_json_path)

    if scanner_shift_df.empty:
        scanner_shift_df = pd.DataFrame(columns=["feature", "group_1", "group_2", "p_value_mannwhitney", "median_group_1", "median_group_2", "abs_median_difference"])
    scanner_shift_df.to_csv(scanner_shift_csv_path, index=False, encoding="utf-8-sig")

    step03_summary = {}
    if Path(cfg.step03_summary_path).exists():
        try:
            step03_summary = load_json(Path(cfg.step03_summary_path))
        except Exception:
            step03_summary = {}

    split_target_balance = {}
    for target_col in feature_dict["target_columns"]:
        if target_col in patient_df.columns:
            balance = patient_df.groupby("split")[target_col].agg(["count", "sum", "mean"]).reset_index()
            balance.columns = ["split", "n_total", "n_positive", "positive_rate"]
            split_target_balance[target_col] = balance.to_dict(orient="records")

    summary = {
        "input_mode": input_mode,
        "input_path": str(input_path),
        "n_input_rows_after_filters": int(df.shape[0]),
        "n_input_columns": int(df.shape[1]),
        "n_conflict_patients_excluded": int(len(conflict_patients)),
        "n_radiomics_columns_detected": int(len(radiomics_cols)),
        "n_case_rows": int(case_df.shape[0]),
        "n_case_columns": int(case_df.shape[1]),
        "n_patient_rows": int(patient_df.shape[0]),
        "n_patient_columns": int(patient_df.shape[1]),
        "n_numeric_model_features": int(len(feature_dict["numeric_feature_columns"])),
        "n_numeric_model_features_no_scanner_covariates": int(len(feature_dict["numeric_feature_columns_no_scanner_covariates"])),
        "n_scanner_covariate_columns": int(len(feature_dict["scanner_covariate_columns"])),
        "n_field_strength_only_candidate_columns": int(len(feature_dict["field_strength_only_candidate_columns"])),
        "n_categorical_model_features": int(len(feature_dict["categorical_feature_columns"])),
        "n_high_missingness_excluded": int(len(feature_dict["high_missingness_excluded_columns"])),
        "patient_split_counts": patient_df["split"].value_counts(dropna=False).to_dict(),
        "patient_stratum_counts": patient_df["stratum_key"].value_counts(dropna=False).to_dict(),
        "target_non_missing_counts": {col: int(patient_df[col].notna().sum()) for col in feature_dict["target_columns"]},
        "split_target_balance": split_target_balance,
        "split_field_strength_balance": build_split_field_strength_balance(patient_df, cfg),
        "field_strength_patient_counts": patient_df[cfg.field_strength_dominant_col].astype(str).fillna("missing").value_counts(dropna=False).to_dict() if cfg.field_strength_dominant_col in patient_df.columns else {},
        "field_strength_lesion_counts": df[cfg.field_strength_label_col].fillna("missing").value_counts(dropna=False).to_dict() if cfg.field_strength_label_col in df.columns else {},
        "scanner_audit_summary": {
            "enabled": scanner_audit.get("enabled", False),
            "available": scanner_audit.get("available", False),
            "n_radiomics_shift_features_tested": scanner_audit.get("n_radiomics_shift_features_tested"),
            "n_radiomics_shift_features_p_lt_0_05": scanner_audit.get("n_radiomics_shift_features_p_lt_0_05"),
            "n_radiomics_shift_features_p_lt_0_0001": scanner_audit.get("n_radiomics_shift_features_p_lt_0_0001"),
        },
        "gpu_available_torch": runtime_info.get("gpu_available_torch"),
        "gpu_name": runtime_info.get("gpu_name"),
        "step03_input_summary": {
            "n_rows_in_step03_input": step03_summary.get("n_rows"),
            "analysis_candidate_final_count": step03_summary.get("analysis_candidate_final_count"),
            "lesion_only_rows": step03_summary.get("lesion_only_rows"),
        },
        "method_notes": {
            "patient_level_split_only": True,
            "patient_label_conflicts_excluded": True,
            "stratification_optionally_includes_field_strength": cfg.include_field_strength_in_stratification,
            "scanner_covariates_explicitly_preserved": True,
            "combat_not_applied_in_stage04_to_avoid_leakage": True,
        },
    }
    summary_path = metadata_dir / f"step04_summary{suffix}.json"
    save_json(summary, summary_path)

    case_csv = processed_dir / f"analysis_ready_step04_case_level{suffix}.csv"
    case_parquet = processed_dir / f"analysis_ready_step04_case_level{suffix}.parquet"
    patient_csv = processed_dir / f"analysis_ready_step04_patient_level{suffix}.csv"
    patient_parquet = processed_dir / f"analysis_ready_step04_patient_level{suffix}.parquet"
    missingness_csv = metadata_dir / f"step04_patient_missingness_report{suffix}.csv"
    splits_csv = metadata_dir / f"step04_patient_splits{suffix}.csv"

    if cfg.save_long_case_table:
        save_df(case_df, case_csv, case_parquet, logger, save_parquet=cfg.save_parquet_if_available)
    if cfg.save_long_patient_table:
        save_df(patient_df, patient_csv, patient_parquet, logger, save_parquet=cfg.save_parquet_if_available)
    missingness_df.to_csv(missingness_csv, index=False, encoding="utf-8-sig")
    if cfg.save_splits:
        splits_df.to_csv(splits_csv, index=False, encoding="utf-8-sig")

    artifacts = {
        "step04_config_json": str(step04_config_path),
        "step04_runtime_environment_json": str(runtime_path),
        "step04_label_integrity_report_json": str(label_report_path),
        "step04_feature_dictionary_json": str(feature_dict_path),
        "step04_summary_json": str(summary_path),
        "step04_scanner_confounding_audit_json": str(scanner_audit_json_path),
        "step04_scanner_radiomics_shift_screening_csv": str(scanner_shift_csv_path),
        "step04_patient_missingness_report_csv": str(missingness_csv),
        "step04_patient_label_conflict_report_csv": str(conflict_report_path),
        "step04_log_file": str(log_file),
    }
    if cfg.save_long_case_table:
        artifacts["analysis_ready_step04_case_level_csv"] = str(case_csv)
    if cfg.save_long_patient_table:
        artifacts["analysis_ready_step04_patient_level_csv"] = str(patient_csv)
    if cfg.save_splits:
        artifacts["step04_patient_splits_csv"] = str(splits_csv)

    # A sensitivity run must not record itself as the step's output: the one-session run once
    # replaced every step-04 artifact path in the pipeline state with its *_onesession file.
    if sensitivity_run:
        logger.info("Sensitivity run (%s): pipeline state left pointing at the primary outputs.", suffix)
    else:
        state.mark_step_done(STEP_NAME, artifacts=artifacts)
        state.add_note("Step 04 scanner-aware patient-level feature engineering and modeling prep completed successfully.")
    logger.info("Saved Step 04 summary to: %s", summary_path)
    logger.info("Completed %s successfully.", STEP_NAME)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()