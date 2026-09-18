from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================================================
# 1) SETTINGS
# =========================================================

PROJECT_ROOT = Path("./bcbm_project").resolve()
STEP_NAME = "step_02_data_audit"

MAX_UNIQUE_SAMPLES = 25
HIGH_MISSING_THRESHOLD = 0.50
FULL_UNIQUES_IF_LEQ = 30

# FIX: matched_mask_file يُقدَّم على image_path لأنه يحتوي على اسم القناع الفعلي
CASE_SOURCE_PRIORITY = [
    # Prefer explicit corrected identifiers from the updated master file first.
    "case_id",
    "patient_base",
    "visit_index",
    # Fallbacks for reconstruction from filenames / paths.
    "matched_mask_file",
    "FilenamePrefix",
    "ID",
    "image_path",
    "mask_path",
]

IMAGES_MASKS_ROOT_NAME = "BCBM-RadioGenomics_Images_Masks_Dec2024"
NIFTI_EXTENSIONS = {".nii", ".gz"}

FULL_FOLDER_PATTERN = re.compile(r"(BCBM-RadioGenomics-(\d+)-(\d+))", re.IGNORECASE)
CASE_ID_PATTERN = re.compile(r"\b(\d+)-(\d+)\b")

# FIX: استخدام prefix anchoring متسق مع Stage 4+5
RADIOMICS_PREFIXES = [
    "original_", "wavelet_", "log_sigma_", "lbp_",
    "shape_", "firstorder_", "glcm_", "glrlm_",
    "glszm_", "gldm_", "ngtdm_", "diagnostics_",
]

SANITY_PASS_THRESHOLD = 0.95


# =========================================================
# 2) CONFIG
# =========================================================

@dataclass
class Step02Config:
    project_root: str = str(PROJECT_ROOT)
    step_name: str = STEP_NAME
    max_unique_samples: int = MAX_UNIQUE_SAMPLES
    high_missing_threshold: float = HIGH_MISSING_THRESHOLD
    full_uniques_if_leq: int = FULL_UNIQUES_IF_LEQ
    case_source_priority: List[str] = None
    images_masks_root_name: str = IMAGES_MASKS_ROOT_NAME
    sanity_pass_threshold: float = SANITY_PASS_THRESHOLD

    def __post_init__(self) -> None:
        if self.case_source_priority is None:
            self.case_source_priority = CASE_SOURCE_PRIORITY.copy()


# =========================================================
# 3) STATE MANAGER
# =========================================================

class StateManager:
    def __init__(self, state_path: Path):
        self.state_path = state_path
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if self.state_path.exists():
            with open(self.state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "project_initialized": False,
            "steps_completed": [],
            "artifacts": {},
            "notes": [],
            "last_updated": None,
        }

    def save(self) -> None:
        self.state["last_updated"] = datetime.utcnow().isoformat()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
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
        self.state.setdefault("notes", []).append({
            "time": datetime.utcnow().isoformat(),
            "note": note,
        })
        self.save()


# =========================================================
# 4) IO / LOGGING
# =========================================================

def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger(STEP_NAME)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def detect_excel_engine(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        return "openpyxl"
    if suffix == ".xls":
        return "xlrd"
    raise ValueError(f"Unsupported Excel extension: {suffix}")


def safe_read_excel(file_path: str, sheet_name=0) -> pd.DataFrame:
    # FIX I-17: Auto-detect active sheet — if only one sheet exists, use it
    engine = detect_excel_engine(file_path)
    xl = pd.ExcelFile(file_path, engine=engine)
    sheet = xl.sheet_names[0] if len(xl.sheet_names) == 1 else sheet_name
    return pd.read_excel(file_path, sheet_name=sheet, engine=engine)


# =========================================================
# 5) HELPERS
# =========================================================

def normalize_colname(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[\s\-/\\]+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def normalize_text(value: Any) -> str:
    s = str(value).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def missing_ratio(series: pd.Series) -> float:
    return float(series.isna().mean())


def non_null_unique_count(series: pd.Series) -> int:
    return int(series.dropna().nunique())


def sample_unique_values(series: pd.Series, max_samples: int = MAX_UNIQUE_SAMPLES) -> List[str]:
    vals = series.dropna().astype(str).str.strip()
    vals = vals[vals != ""].unique().tolist()
    return sorted(vals)[:max_samples]


def maybe_binary_like(series: pd.Series) -> bool:
    vals = {
        str(v).strip().lower()
        for v in series.dropna().unique().tolist()
        if str(v).strip() != ""
    }
    if not vals:
        return False
    binary_tokens = {
        "0", "1", "yes", "no", "positive", "negative", "pos", "neg",
        "true", "false", "er+", "er-", "pr+", "pr-", "her2+", "her2-", "+", "-",
    }
    return len(vals) <= 6 and vals.issubset(binary_tokens)


def classify_column_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "text_or_categorical"


def score_column_name(col: str, keywords: List[str]) -> int:
    c = normalize_colname(col)
    score = 0
    for kw in keywords:
        kw_norm = normalize_colname(kw)
        if c == kw_norm:
            score += 100
        elif kw_norm in c:
            score += 20
    return score


def find_best_matching_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    scored = [(score_column_name(c, candidates), c) for c in columns]
    scored = [x for x in scored if x[0] > 0]
    if not scored:
        return None
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][1]


# FIX: prefix anchoring متسق مع Stage 4+5
def infer_radiomics_columns(df: pd.DataFrame) -> List[str]:
    out: List[str] = []
    for col in df.columns:
        col_norm = normalize_colname(col) + "_"
        if is_numeric_series(df[col]) and any(col_norm.startswith(pfx) for pfx in RADIOMICS_PREFIXES):
            out.append(col)
    return out


def infer_metadata_columns(df: pd.DataFrame, radiomics_cols: List[str]) -> List[str]:
    return [c for c in df.columns if c not in radiomics_cols]


# =========================================================
# 6) CASE / IDENTIFIER EXTRACTION
# =========================================================

def normalize_explicit_case_id(case_value: Any) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[int]]:
    """
    Accept either:
      - BCBM-RadioGenomics-12-0
      - 12-0
    and always return:
      folder_name = BCBM-RadioGenomics-12-0
      case_id     = 12-0
      patient_base= 12
      visit_index = 0
    """
    if pd.isna(case_value):
        return None, None, None, None

    s = str(case_value).strip().replace("/", "\\")
    if not s:
        return None, None, None, None

    m = FULL_FOLDER_PATTERN.search(s)
    if m:
        folder_name = m.group(1)
        patient_base = int(m.group(2))
        visit_index = int(m.group(3))
        case_id = f"{patient_base}-{visit_index}"
        return folder_name, case_id, patient_base, visit_index

    m = CASE_ID_PATTERN.search(s)
    if m:
        patient_base = int(m.group(1))
        visit_index = int(m.group(2))
        case_id = f"{patient_base}-{visit_index}"
        folder_name = f"BCBM-RadioGenomics-{case_id}"
        return folder_name, case_id, patient_base, visit_index

    return None, None, None, None


def extract_case_fields_from_text(text: Any) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[int]]:
    if pd.isna(text):
        return None, None, None, None
    text = str(text).replace("/", "\\").strip()

    m = FULL_FOLDER_PATTERN.search(text)
    if m:
        folder_name  = m.group(1)
        patient_base = int(m.group(2))
        visit_index  = int(m.group(3))
        case_id      = f"{patient_base}-{visit_index}"
        return folder_name, case_id, patient_base, visit_index

    m2 = CASE_ID_PATTERN.search(text)
    if m2:
        patient_base = int(m2.group(1))
        visit_index  = int(m2.group(2))
        case_id      = f"{patient_base}-{visit_index}"
        folder_name  = f"BCBM-RadioGenomics-{case_id}"
        return folder_name, case_id, patient_base, visit_index

    return None, None, None, None


def _safe_int_like(value: Any) -> Optional[int]:
    if pd.isna(value):
        return None
    try:
        return int(float(value))
    except Exception:
        s = str(value).strip()
        if not s:
            return None
        m = re.search(r"-?\d+", s)
        if m:
            return int(m.group(0))
    return None


def extract_case_fields_row(row: pd.Series, source_priority: List[str]) -> pd.Series:
    # 1) Prefer explicit corrected identifiers already present in the updated master file.
    row_has_case = "case_id" in row.index and pd.notna(row.get("case_id"))
    row_has_patient = "patient_base" in row.index and pd.notna(row.get("patient_base"))
    row_has_visit = "visit_index" in row.index and pd.notna(row.get("visit_index"))

    if row_has_case:
        folder_name, case_id, patient_base, visit_index = normalize_explicit_case_id(row.get("case_id"))
        if case_id is not None:
            patient_base_explicit = _safe_int_like(row.get("patient_base")) if row_has_patient else patient_base
            visit_index_explicit = _safe_int_like(row.get("visit_index")) if row_has_visit else visit_index

            if patient_base_explicit is not None and visit_index_explicit is not None:
                patient_base = patient_base_explicit
                visit_index = visit_index_explicit
                case_id = f"{patient_base}-{visit_index}"
                folder_name = f"BCBM-RadioGenomics-{case_id}"

            return pd.Series({
                "case_source_column": "case_id_explicit",
                "case_folder_name":   folder_name,
                "case_id":            case_id,
                "patient_base":       patient_base,
                "visit_index":        visit_index,
            })

    if row_has_patient and row_has_visit:
        patient_base = _safe_int_like(row.get("patient_base"))
        visit_index = _safe_int_like(row.get("visit_index"))
        if patient_base is not None and visit_index is not None:
            case_id = f"{patient_base}-{visit_index}"
            folder_name = f"BCBM-RadioGenomics-{case_id}"
            return pd.Series({
                "case_source_column": "patient_visit_explicit",
                "case_folder_name":   folder_name,
                "case_id":            case_id,
                "patient_base":       patient_base,
                "visit_index":        visit_index,
            })

    # 2) Fallback to filename/path based inference.
    for col in source_priority:
        if col in {"case_id", "patient_base", "visit_index"}:
            continue
        if col in row.index:
            folder_name, case_id, patient_base, visit_index = extract_case_fields_from_text(row[col])
            if case_id is not None:
                return pd.Series({
                    "case_source_column": col,
                    "case_folder_name":   folder_name,
                    "case_id":            case_id,
                    "patient_base":       patient_base,
                    "visit_index":        visit_index,
                })
    return pd.Series({
        "case_source_column": None,
        "case_folder_name":   None,
        "case_id":            None,
        "patient_base":       None,
        "visit_index":        None,
    })


def add_case_patient_columns(df: pd.DataFrame, source_priority: List[str]) -> pd.DataFrame:
    id_cols = ["case_source_column", "case_folder_name", "case_id", "patient_base", "visit_index"]

    backup_map = {
        "case_id": "case_id_original_input",
        "patient_base": "patient_base_original_input",
        "visit_index": "visit_index_original_input",
    }
    for src, dst in backup_map.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    df = df.drop(columns=[c for c in id_cols if c in df.columns])
    extracted = df.apply(lambda row: extract_case_fields_row(row, source_priority), axis=1)
    return pd.concat([df, extracted], axis=1)


# =========================================================
# 7) PATH RECONSTRUCTION / QA
# =========================================================

def reconstruct_absolute_path(
    relative_path: Any,
    raw_images_masks_dir: Path,
    root_name: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[str]:
    if pd.isna(relative_path):
        return None
    rel = str(relative_path).strip().replace("/", "\\")
    if not rel:
        return None
    parts = [p for p in rel.split("\\") if p.strip()]
    if not parts:
        return None
    if parts[0].lower() == root_name.lower():
        parts = parts[1:]
    abs_path = raw_images_masks_dir.joinpath(*parts).resolve()
    suffixes = {s.lower() for s in abs_path.suffixes}
    if not suffixes.intersection(NIFTI_EXTENSIONS) and logger is not None:
        logger.warning("Unexpected file extension for reconstructed path: %s", abs_path)
    return str(abs_path)


def path_exists(path_str: Any) -> bool:
    if path_str is None or (isinstance(path_str, float) and pd.isna(path_str)):
        return False
    try:
        return Path(path_str).exists()
    except (OSError, ValueError):
        return False


def add_reconstructed_paths(
    df: pd.DataFrame,
    raw_images_masks_dir: Path,
    root_name: str,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    if "image_path" in df.columns:
        df["image_abs_path"]   = df["image_path"].apply(lambda x: reconstruct_absolute_path(x, raw_images_masks_dir, root_name, logger))
        df["image_abs_exists"] = df["image_abs_path"].apply(path_exists)
    if "mask_path" in df.columns:
        df["mask_abs_path"]   = df["mask_path"].apply(lambda x: reconstruct_absolute_path(x, raw_images_masks_dir, root_name, logger))
        df["mask_abs_exists"] = df["mask_abs_path"].apply(path_exists)
    return df


def case_id_from_path_text(path_text: Any) -> Optional[str]:
    _, case_id, _, _ = extract_case_fields_from_text(path_text)
    return case_id


def build_image_mask_alignment_report(df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    if "case_id" not in df.columns:
        return {"alignment_checked": False, "reason": "case_id_missing"}, pd.DataFrame()

    for idx, row in df.iterrows():
        image_case     = case_id_from_path_text(row.get("image_path")) if "image_path" in df.columns else None
        mask_case      = case_id_from_path_text(row.get("mask_path"))  if "mask_path"  in df.columns else None
        canonical_case = row.get("case_id")

        image_ok = (image_case == canonical_case) if image_case is not None and pd.notna(canonical_case) else None
        mask_ok  = (mask_case  == canonical_case) if mask_case  is not None and pd.notna(canonical_case) else None
        pair_ok  = None
        if image_case is not None and mask_case is not None:
            pair_ok = (image_case == mask_case == canonical_case)

        rows.append({
            "row_index":                    int(idx),
            "case_id":                      canonical_case,
            "image_case_id_from_path":      image_case,
            "mask_case_id_from_path":       mask_case,
            "image_case_match":             image_ok,
            "mask_case_match":              mask_ok,
            "image_mask_case_alignment":    pair_ok,
            "image_abs_path":               row.get("image_abs_path"),
            "mask_abs_path":                row.get("mask_abs_path"),
        })

    align_df     = pd.DataFrame(rows)
    checked_pair = align_df["image_mask_case_alignment"].notna().sum()
    full_match   = int((align_df["image_mask_case_alignment"] == True).sum())
    mismatch     = int((align_df["image_mask_case_alignment"] == False).sum())

    report = {
        "alignment_checked":                    True,
        "rows_total":                           int(len(align_df)),
        "rows_with_checkable_image_mask_pair":  int(checked_pair),
        "full_alignment_count":                 full_match,
        "full_alignment_ratio":                 float(full_match / checked_pair) if checked_pair > 0 else None,
        "mismatch_count":                       mismatch,
        "mismatch_example_case_ids":            align_df.loc[align_df["image_mask_case_alignment"] == False, "case_id"].dropna().astype(str).unique().tolist()[:20],
    }
    return report, align_df


def build_qa_flags(df: pd.DataFrame, alignment_df: pd.DataFrame) -> pd.DataFrame:
    qa = pd.DataFrame(index=df.index)
    qa["qa_missing_case_id"]      = df["case_id"].isna().astype(int)      if "case_id"      in df.columns else 1
    qa["qa_missing_patient_base"] = df["patient_base"].isna().astype(int)  if "patient_base" in df.columns else 1
    qa["qa_missing_visit_index"]  = df["visit_index"].isna().astype(int)   if "visit_index"  in df.columns else 1
    qa["qa_missing_image_file"]   = (~df["image_abs_exists"]).astype(int)  if "image_abs_exists" in df.columns else 1
    qa["qa_missing_mask_file"]    = (~df["mask_abs_exists"]).astype(int)   if "mask_abs_exists"  in df.columns else 1

    if not alignment_df.empty:
        aligned   = alignment_df.set_index("row_index")
        misalign  = aligned["image_mask_case_alignment"].map(lambda x: int(x is False)).reindex(df.index).fillna(0).astype(int)
        qa["qa_image_mask_case_mismatch"] = misalign
    else:
        qa["qa_image_mask_case_mismatch"] = 0

    qa["qa_any_issue"] = (qa.sum(axis=1) > 0).astype(int)
    return qa


def build_dataset_sanity_score(
    df: pd.DataFrame,
    alignment_report: Dict[str, Any],
    integrity_report: Dict[str, Any],
) -> Dict[str, Any]:
    checks = []
    if "image_abs_exists" in df.columns:
        checks.append(bool(df["image_abs_exists"].all()))
    if "mask_abs_exists" in df.columns:
        checks.append(bool(df["mask_abs_exists"].all()))
    checks.append(integrity_report.get("missing_case_id_rows",    1) == 0)
    checks.append(integrity_report.get("missing_patient_base_rows",1) == 0)
    checks.append(integrity_report.get("missing_visit_index_rows", 1) == 0)
    checks.append(integrity_report.get("n_case_to_patient_conflicts", 1) == 0)
    checks.append(integrity_report.get("n_case_to_visit_conflicts",   1) == 0)
    if alignment_report.get("alignment_checked"):
        checks.append(alignment_report.get("mismatch_count", 1) == 0)

    passed = int(sum(checks))
    total  = int(len(checks))
    score  = float(passed / total) if total > 0 else 0.0
    return {
        "checks_total":  total,
        "checks_passed": passed,
        "sanity_score":  score,
        "sanity_status": "pass" if score >= SANITY_PASS_THRESHOLD else "warning",
    }


# =========================================================
# 8) AUDIT / VALIDATION REPORTS
# =========================================================

def build_column_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for col in df.columns:
        s    = df[col]
        row: Dict[str, Any] = {
            "column_name":          col,
            "normalized_name":      normalize_colname(col),
            "dtype":                str(s.dtype),
            "inferred_type":        classify_column_type(s),
            "n_missing":            int(s.isna().sum()),
            "missing_ratio":        round(float(s.isna().mean()), 6),
            "n_unique_non_null":    non_null_unique_count(s),
            "n_unique_shown":       min(non_null_unique_count(s), MAX_UNIQUE_SAMPLES),
            "is_numeric":           bool(is_numeric_series(s)),
            "is_binary_like":       bool(maybe_binary_like(s)),
        }
        if classify_column_type(s) == "text_or_categorical":
            row["sample_unique_values"] = " | ".join(sample_unique_values(s, MAX_UNIQUE_SAMPLES))
            row.update({"min": "", "max": "", "mean": "", "median": "", "std": ""})
        else:
            row["sample_unique_values"] = ""
            try:
                vals = s.dropna().values.astype(float)
                row["min"]    = float(np.nanmin(vals))
                row["max"]    = float(np.nanmax(vals))
                row["mean"]   = float(np.nanmean(vals))
                row["median"] = float(np.nanmedian(vals))
                row["std"]    = float(np.nanstd(vals))
            except Exception:
                row.update({"min": np.nan, "max": np.nan, "mean": np.nan, "median": np.nan, "std": np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def extract_small_uniques_catalog(df: pd.DataFrame) -> Dict[str, List[str]]:
    catalog: Dict[str, List[str]] = {}
    for col in df.columns:
        if not is_numeric_series(df[col]):
            vals = df[col].dropna().astype(str).str.strip()
            vals = vals[vals != ""].unique().tolist()
            vals = sorted(vals)
            if len(vals) <= FULL_UNIQUES_IF_LEQ:
                catalog[col] = vals
    return catalog


def detect_schema(df: pd.DataFrame, project_cfg: Dict[str, Any]) -> Dict[str, Any]:
    columns              = list(df.columns)
    original_patient_col = find_best_matching_column(columns, project_cfg.get("patient_id_column_candidates", []))
    lesion_name_col      = find_best_matching_column(columns, project_cfg.get("lesion_name_column_candidates", []))
    label_candidates     = project_cfg.get("label_column_candidates", {})
    er_col   = find_best_matching_column(columns, label_candidates.get("ER",   []))
    pr_col   = find_best_matching_column(columns, label_candidates.get("PR",   []))
    her2_col = find_best_matching_column(columns, label_candidates.get("HER2", []))

    radiomics_cols  = infer_radiomics_columns(df)
    metadata_cols   = infer_metadata_columns(df, radiomics_cols)
    numeric_cols    = [c for c in df.columns if is_numeric_series(df[c])]
    text_cols       = [c for c in df.columns if not is_numeric_series(df[c])]
    high_missing    = [c for c in df.columns if missing_ratio(df[c]) >= HIGH_MISSING_THRESHOLD]
    binary_like     = [c for c in df.columns if maybe_binary_like(df[c])]

    detected_patient = "patient_base" if "patient_base" in df.columns and df["patient_base"].notna().any() else original_patient_col
    detected_case    = "case_id"      if "case_id"      in df.columns and df["case_id"].notna().any()      else None

    return {
        "detected_patient_id_column":          detected_patient,
        "detected_case_id_column":             detected_case,
        "detected_original_patient_like_column": original_patient_col,
        "detected_segmentation_name_column":   lesion_name_col,
        "detected_label_columns":              {"ER": er_col, "PR": pr_col, "HER2": her2_col},
        "n_total_columns":                     len(df.columns),
        "n_numeric_columns":                   len(numeric_cols),
        "n_text_or_categorical_columns":       len(text_cols),
        "n_radiomics_like_columns":            len(radiomics_cols),
        "n_metadata_like_columns":             len(metadata_cols),
        "high_missing_columns":                high_missing,
        "binary_like_columns":                 binary_like,
        "radiomics_like_columns":              radiomics_cols,
        "metadata_like_columns":               metadata_cols,
    }


def build_dataset_overview(df: pd.DataFrame, schema: Dict[str, Any]) -> Dict[str, Any]:
    overview: Dict[str, Any] = {
        "n_rows":          int(df.shape[0]),
        "n_columns":       int(df.shape[1]),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 3),
    }
    patient_col = schema.get("detected_patient_id_column")
    case_col    = schema.get("detected_case_id_column")
    lesion_col  = schema.get("detected_segmentation_name_column")
    overview["n_unique_patients"]          = int(df[patient_col].nunique(dropna=True)) if patient_col and patient_col in df.columns else None
    overview["n_unique_cases"]             = int(df[case_col].nunique(dropna=True))    if case_col    and case_col    in df.columns else None
    overview["n_unique_segmentation_names"] = int(df[lesion_col].nunique(dropna=True)) if lesion_col  and lesion_col  in df.columns else None
    if "image_abs_exists" in df.columns:
        overview["image_abs_exists_count"]  = int(df["image_abs_exists"].sum())
        overview["image_abs_missing_count"] = int((~df["image_abs_exists"]).sum())
    if "mask_abs_exists" in df.columns:
        overview["mask_abs_exists_count"]  = int(df["mask_abs_exists"].sum())
        overview["mask_abs_missing_count"] = int((~df["mask_abs_exists"]).sum())
    return overview


def build_task_readiness_report(df: pd.DataFrame, schema: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for label_name, col in schema.get("detected_label_columns", {}).items():
        if col is None or col not in df.columns:
            result[label_name] = {"column_found": False, "non_missing_count": 0, "unique_values": []}
            continue
        vals   = df[col].dropna().astype(str).str.strip()
        vals   = vals[vals != ""]
        uniques = sorted(vals.unique().tolist())
        result[label_name] = {
            "column_found":     True,
            "column_name":      col,
            "non_missing_count": int(vals.shape[0]),
            "n_unique_values":  len(uniques),
            "unique_values":    uniques[:50],
        }
    return result


def build_case_extraction_report(df: pd.DataFrame) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "n_rows":               int(df.shape[0]),
        "n_rows_with_case_id":  int(df["case_id"].notna().sum())  if "case_id"      in df.columns else 0,
        "n_rows_without_case_id": int(df["case_id"].isna().sum()) if "case_id"      in df.columns else 0,
        "n_unique_case_id":     int(df["case_id"].nunique(dropna=True))      if "case_id"      in df.columns else 0,
        "n_unique_patient_base": int(df["patient_base"].nunique(dropna=True)) if "patient_base" in df.columns else 0,
        "case_source_column_counts": df["case_source_column"].value_counts(dropna=False).to_dict() if "case_source_column" in df.columns else {},
    }
    if "image_abs_exists" in df.columns:
        report["image_abs_exists_count"]  = int(df["image_abs_exists"].sum())
        report["image_abs_missing_count"] = int((~df["image_abs_exists"]).sum())
    if "mask_abs_exists" in df.columns:
        report["mask_abs_exists_count"]  = int(df["mask_abs_exists"].sum())
        report["mask_abs_missing_count"] = int((~df["mask_abs_exists"]).sum())
    return report


def build_identifier_integrity_report(df: pd.DataFrame) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    if not {"case_id", "patient_base", "visit_index"}.issubset(df.columns):
        report["identifier_columns_present"] = False
        return report
    report["identifier_columns_present"] = True

    missing_case    = int(df["case_id"].isna().sum())
    missing_patient = int(df["patient_base"].isna().sum())
    missing_visit   = int(df["visit_index"].isna().sum())

    case_to_patient_conflicts = df.dropna(subset=["case_id", "patient_base"]).groupby("case_id")["patient_base"].nunique()
    case_to_patient_conflicts = case_to_patient_conflicts[case_to_patient_conflicts > 1]
    case_to_visit_conflicts   = df.dropna(subset=["case_id", "visit_index"]).groupby("case_id")["visit_index"].nunique()
    case_to_visit_conflicts   = case_to_visit_conflicts[case_to_visit_conflicts > 1]
    patient_case_counts       = df.dropna(subset=["patient_base", "case_id"]).groupby("patient_base")["case_id"].nunique()

    report.update({
        "missing_case_id_rows":                 missing_case,
        "missing_patient_base_rows":            missing_patient,
        "missing_visit_index_rows":             missing_visit,
        "n_case_to_patient_conflicts":          int(case_to_patient_conflicts.shape[0]),
        "n_case_to_visit_conflicts":            int(case_to_visit_conflicts.shape[0]),
        "case_to_patient_conflict_case_ids":    sorted(case_to_patient_conflicts.index.tolist())[:20],
        "case_to_patient_conflict_examples":    case_to_patient_conflicts.head(20).to_dict(),
        "case_to_visit_conflict_case_ids":      sorted(case_to_visit_conflicts.index.tolist())[:20],
        "case_to_visit_conflict_examples":      case_to_visit_conflicts.head(20).to_dict(),
        "patients_with_multiple_cases":         int((patient_case_counts > 1).sum()),
        "max_cases_per_patient":                int(patient_case_counts.max()) if not patient_case_counts.empty else 0,
    })
    return report


def enforce_identifier_integrity(report: Dict[str, Any], max_missing_fraction: float = 0.05,
                                   logger=None) -> None:
    # FIX I-05: Tolerant enforcement — warn on small missingness, raise only on severe issues
    if not report.get("identifier_columns_present", False):
        raise ValueError("Identifier columns are missing after extraction.")

    n_rows = max(report.get("rows_total", 1), 1)
    for field_name, label in [
        ("missing_case_id_rows", "case_id"),
        ("missing_patient_base_rows", "patient_base"),
        ("missing_visit_index_rows", "visit_index"),
    ]:
        count = report.get(field_name, 0)
        fraction = count / n_rows
        if fraction > max_missing_fraction:
            raise ValueError(
                f"{label}: {count} rows ({fraction:.1%}) missing identifiers — "
                f"exceeds tolerance of {max_missing_fraction:.0%}. "
                "Check case extraction logic or set max_missing_fraction higher."
            )
        elif count > 0 and logger:
            logger.warning(
                "%s: %d rows (%s) missing — within tolerance (%s), continuing.",
                label, count, f"{fraction:.1%}", f"{max_missing_fraction:.0%}"
            )

    if report.get("n_case_to_patient_conflicts", 0) > 0:
        raise ValueError(
            f"Found {report['n_case_to_patient_conflicts']} case_id values mapped to multiple "
            f"patient_base values. Sample: {report.get('case_to_patient_conflict_case_ids', [])}"
        )
    if report.get("n_case_to_visit_conflicts", 0) > 0:
        raise ValueError(
            f"Found {report['n_case_to_visit_conflicts']} case_id values mapped to multiple "
            f"visit_index values. Sample: {report.get('case_to_visit_conflict_case_ids', [])}"
        )


# =========================================================
# 9) MAIN
# =========================================================

def main() -> None:
    cfg = Step02Config()

    metadata_dir     = PROJECT_ROOT / "metadata"
    logs_dir         = PROJECT_ROOT / "logs"
    checkpoints_dir  = PROJECT_ROOT / "checkpoints"
    interim_dir      = PROJECT_ROOT / "data" / "interim"
    raw_dir          = PROJECT_ROOT / "data" / "raw"
    raw_images_masks_dir = raw_dir / cfg.images_masks_root_name

    for p in [metadata_dir, logs_dir, checkpoints_dir, interim_dir]:
        p.mkdir(parents=True, exist_ok=True)

    state_path = checkpoints_dir / "pipeline_state.json"
    log_file   = logs_dir / f"{STEP_NAME}.log"
    logger     = setup_logger(log_file)
    state      = StateManager(state_path)

    logger.info("=" * 80)
    logger.info("Starting dataset audit and canonicalization")

    if not state.is_step_done("step_01_project_initialization"):
        raise RuntimeError("Step 01 must be completed before Step 02.")

    config_path = metadata_dir / "project_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    project_cfg = load_json(config_path)

    input_xlsx = project_cfg.get("input_xlsx_project_copy")
    if not input_xlsx:
        raise ValueError("input_xlsx_project_copy not found in project_config.json")
    input_xlsx = Path(input_xlsx)
    if not input_xlsx.exists():
        raise FileNotFoundError(f"Dataset file not found: {input_xlsx}")

    images_masks_dir_exists = raw_images_masks_dir.exists()
    if not images_masks_dir_exists:
        logger.warning("Images/masks root folder NOT found: %s", raw_images_masks_dir)
    else:
        logger.info("Images/masks root folder found: %s", raw_images_masks_dir)

    logger.info("Reading dataset from: %s", input_xlsx)
    df = safe_read_excel(str(input_xlsx))
    logger.info("Dataset loaded with shape: %s", df.shape)

    logger.info("Extracting case/patient identity (priority: %s)...", cfg.case_source_priority)
    df = add_case_patient_columns(df, cfg.case_source_priority)
    if "case_source_column" in df.columns:
        logger.info("Identifier extraction source counts: %s", df["case_source_column"].value_counts(dropna=False).to_dict())

    logger.info("Reconstructing absolute image/mask paths...")
    df = add_reconstructed_paths(df, raw_images_masks_dir, cfg.images_masks_root_name, logger)

    logger.info("Checking identifier integrity...")
    identifier_integrity      = build_identifier_integrity_report(df)
    identifier_integrity_path = metadata_dir / "identifier_integrity_report.json"
    save_json(identifier_integrity, identifier_integrity_path)
    enforce_identifier_integrity(identifier_integrity, logger=logger)

    logger.info("Checking image/mask-case alignment...")
    alignment_report, alignment_df = build_image_mask_alignment_report(df)
    alignment_report_path  = metadata_dir / "image_mask_alignment_report.json"
    alignment_preview_path = metadata_dir / "image_mask_alignment_preview.csv"
    save_json(alignment_report, alignment_report_path)
    alignment_df.head(500).to_csv(alignment_preview_path, index=False, encoding="utf-8-sig")

    logger.info("Building QA flags...")
    qa_flags_df = build_qa_flags(df, alignment_df)
    df = pd.concat([df, qa_flags_df], axis=1)
    qa_cols      = [c for c in df.columns if c.startswith("qa_")] + [c for c in ["case_id", "patient_base", "visit_index"] if c in df.columns]
    qa_flags_path = metadata_dir / "dataset_qa_flags_preview.csv"
    df[qa_cols].head(500).to_csv(qa_flags_path, index=False, encoding="utf-8-sig")

    logger.info("Building column summary...")
    column_summary_df   = build_column_summary(df)
    column_summary_csv  = interim_dir / "column_summary.csv"
    column_summary_xlsx = interim_dir / "column_summary.xlsx"
    column_summary_df.to_csv(column_summary_csv, index=False, encoding="utf-8-sig")
    column_summary_df.to_excel(column_summary_xlsx, index=False)

    logger.info("Building small-uniques catalog...")
    small_uniques_catalog = extract_small_uniques_catalog(df)
    small_uniques_path    = metadata_dir / "small_uniques_catalog.json"
    save_json(small_uniques_catalog, small_uniques_path)

    logger.info("Detecting schema...")
    schema      = detect_schema(df, project_cfg)
    schema_path = metadata_dir / "detected_schema.json"
    save_json(schema, schema_path)

    overview      = build_dataset_overview(df, schema)
    overview_path = metadata_dir / "dataset_overview.json"
    save_json(overview, overview_path)

    task_readiness      = build_task_readiness_report(df, schema)
    task_readiness_path = metadata_dir / "task_readiness_report.json"
    save_json(task_readiness, task_readiness_path)

    case_extraction_report = build_case_extraction_report(df)
    case_extraction_path   = metadata_dir / "case_extraction_report.json"
    save_json(case_extraction_report, case_extraction_path)

    sanity_report      = build_dataset_sanity_score(df, alignment_report, identifier_integrity)
    sanity_report_path = metadata_dir / "dataset_sanity_report.json"
    save_json(sanity_report, sanity_report_path)

    logger.info("Saving canonical CSV...")
    canonical_csv_path = interim_dir / "tabular_raw_canonical.csv"
    df.to_csv(canonical_csv_path, index=False, encoding="utf-8-sig")

    case_preview_cols = [c for c in [
        "FilenamePrefix", "ID", "image_path", "mask_path", "matched_mask_file",
        "case_source_column", "case_folder_name", "case_id", "patient_base", "visit_index",
        "image_abs_path", "image_abs_exists", "mask_abs_path", "mask_abs_exists",
    ] if c in df.columns]
    df[case_preview_cols].drop_duplicates().head(300).to_csv(metadata_dir / "case_preview.csv", index=False, encoding="utf-8-sig")

    if "Segmentation_Name" in df.columns:
        df[["Segmentation_Name"]].drop_duplicates().sort_values("Segmentation_Name").to_csv(metadata_dir / "segmentation_preview.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["Segmentation_Name"]).to_csv(metadata_dir / "segmentation_preview.csv", index=False, encoding="utf-8-sig")

    path_cols = [c for c in ["image_path","image_abs_path","image_abs_exists","mask_path","mask_abs_path","mask_abs_exists"] if c in df.columns]
    df[path_cols].drop_duplicates().head(300).to_csv(metadata_dir / "path_preview.csv", index=False, encoding="utf-8-sig")

    audit_text_path = metadata_dir / "dataset_audit_summary.txt"
    with open(audit_text_path, "w", encoding="utf-8") as f:
        f.write("DATASET AUDIT SUMMARY\n" + "=" * 80 + "\n\n")
        f.write(f"Rows: {overview['n_rows']}\nColumns: {overview['n_columns']}\nMemory (MB): {overview['memory_usage_mb']}\n\n")
        f.write(f"Images/Masks root: {raw_images_masks_dir}\nExists on disk: {images_masks_dir_exists}\n\n")
        f.write("Detected schema:\n")
        for k, v in schema.items():
            if not isinstance(v, list) or len(v) <= 5:
                f.write(f"  {k}: {v}\n")
        f.write("\nSanity report:\n")
        for k, v in sanity_report.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nTask readiness:\n")
        for label_name, info in task_readiness.items():
            f.write(f"  {label_name}: {info}\n")

    step02_config_path = metadata_dir / "step02_config.json"
    save_json(asdict(cfg), step02_config_path)

    state.mark_step_done(STEP_NAME, artifacts={
        "step02_config_json":               str(step02_config_path),
        "column_summary_csv":               str(column_summary_csv),
        "column_summary_xlsx":              str(column_summary_xlsx),
        "small_uniques_catalog":            str(small_uniques_path),
        "detected_schema":                  str(schema_path),
        "dataset_overview":                 str(overview_path),
        "task_readiness_report":            str(task_readiness_path),
        "case_extraction_report":           str(case_extraction_path),
        "identifier_integrity_report":      str(identifier_integrity_path),
        "image_mask_alignment_report":      str(alignment_report_path),
        "image_mask_alignment_preview_csv": str(alignment_preview_path),
        "dataset_sanity_report":            str(sanity_report_path),
        "dataset_qa_flags_preview_csv":     str(qa_flags_path),
        "tabular_raw_canonical_csv":        str(canonical_csv_path),
        "dataset_audit_summary_txt":        str(audit_text_path),
        "step_02_log_file":                 str(log_file),
    })
    state.add_note("Step 02 dataset audit and canonicalization completed successfully.")
    logger.info("Step 02 completed successfully.")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
