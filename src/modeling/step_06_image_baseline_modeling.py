from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import warnings
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path("./bcbm_project").resolve()
STEP_NAME = "step_06_image_baseline_modeling"

INPUT_STEP03_LESION = PROJECT_ROOT / "data" / "processed" / "analysis_ready_step03_lesion_only.csv"
INPUT_STEP04_PATIENT = PROJECT_ROOT / "data" / "processed" / "analysis_ready_step04_patient_level.csv"
INPUT_STEP05_SUMMARY = PROJECT_ROOT / "metadata" / "step05master_summary.json"
INPUT_STEP05_RESULTS = PROJECT_ROOT / "reports" / "tables" / "step05master_model_results.csv"
INPUT_FEATURE_DICT = PROJECT_ROOT / "metadata" / "step04_feature_dictionary.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
METADATA_DIR = PROJECT_ROOT / "metadata"
LOGS_DIR = PROJECT_ROOT / "logs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

STEP06_FIGURES_DIR = FIGURES_DIR / "step06_image_benchmark"
STEP06_TABLES_DIR = TABLES_DIR / "step06_image_benchmark"

TARGET_COLUMNS = ["target_er", "target_pr", "target_her2"]
TARGET_DISPLAY_NAMES = {"target_er": "ER", "target_pr": "PR", "target_her2": "HER2"}
MODALITY_VARIANTS = ["image_only", "radiomics_only", "scanner_only", "image_plus_radiomics", "hybrid_all"]
MODEL_DISPLAY_NAMES = {
    "logistic": "Logistic Regression",
    "elastic_net": "Elastic Net Logistic",
    "svm_linear": "Linear SVM",
    "svm_rbf": "RBF SVM",
}

GLOBAL_SEED = 42
PREDICTION_THRESHOLD_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
SAVE_PLOTS = True
SAVE_PUBLICATION_PACKAGE = True
IMAGE_HIST_BINS = 12
IMAGE_PCA_COMPONENTS = 12
MIN_REQUIRED_TRAIN_ROWS = 10
MIN_REQUIRED_VALID_ROWS = 4
MIN_REQUIRED_TEST_ROWS = 4
MIN_CLASS_COUNT_TRAIN = 2
MAX_BOOTSTRAP = 1000


# --------------------------------------------
# Utilities
# --------------------------------------------

@dataclass
class Step06Config:
    project_root: str = str(PROJECT_ROOT)
    step_name: str = STEP_NAME
    input_step03_lesion_csv: str = str(INPUT_STEP03_LESION)
    input_step04_patient_csv: str = str(INPUT_STEP04_PATIENT)
    input_step05_summary_json: str = str(INPUT_STEP05_SUMMARY)
    input_step05_results_csv: str = str(INPUT_STEP05_RESULTS)
    input_feature_dict_json: str = str(INPUT_FEATURE_DICT)
    target_columns: List[str] = None
    modality_variants: List[str] = None
    global_seed: int = GLOBAL_SEED
    prediction_threshold_grid: List[float] = None
    image_hist_bins: int = IMAGE_HIST_BINS
    image_pca_components: int = IMAGE_PCA_COMPONENTS
    save_plots: bool = SAVE_PLOTS
    save_publication_package: bool = SAVE_PUBLICATION_PACKAGE
    min_required_train_rows: int = MIN_REQUIRED_TRAIN_ROWS
    min_required_valid_rows: int = MIN_REQUIRED_VALID_ROWS
    min_required_test_rows: int = MIN_REQUIRED_TEST_ROWS
    min_class_count_train: int = MIN_CLASS_COUNT_TRAIN
    max_bootstrap: int = MAX_BOOTSTRAP

    def __post_init__(self):
        if self.target_columns is None:
            self.target_columns = TARGET_COLUMNS.copy()
        if self.modality_variants is None:
            self.modality_variants = MODALITY_VARIANTS.copy()
        if self.prediction_threshold_grid is None:
            self.prediction_threshold_grid = PREDICTION_THRESHOLD_GRID.copy()


class StateManager:
    def __init__(self, state_path: Path):
        self.state_path = state_path
        self.state = self._load_state()

    def _load_state(self):
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"steps_completed": [], "artifacts": {}, "notes": []}

    def save(self):
        self.state_path.write_text(json.dumps(self.state, indent=2, ensure_ascii=False), encoding="utf-8")

    def mark_step_done(self, step_name, artifacts=None):
        if step_name not in self.state["steps_completed"]:
            self.state["steps_completed"].append(step_name)
        if artifacts:
            self.state["artifacts"].update(artifacts)
        self.save()

    def add_note(self, note):
        self.state["notes"].append({"note": note})
        self.save()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], path: Path):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ── Inline cache helpers ──────────────────────────────────────────────────────
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def _is_step_cached(manifest_path: Path, input_files: list, config_hash: str, output_files: list) -> bool:
    if not all(Path(p).exists() for p in output_files):
        return False
    if not manifest_path.exists():
        return False
    try:
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        current = {str(p): _sha256(Path(p)) for p in input_files if Path(p).exists()}
        return m.get("config_hash") == config_hash and m.get("input_sha256") == current
    except Exception:
        return False

def _save_step_manifest(manifest_path: Path, input_files: list, config_hash: str) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({
        "input_sha256": {str(p): _sha256(Path(p)) for p in input_files if Path(p).exists()},
        "config_hash": config_hash,
        "created_at": datetime.utcnow().isoformat(),
    }, indent=2), encoding="utf-8")


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger(STEP_NAME)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    formatter = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    sh = logging.StreamHandler()
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    # FIX I-42: Session boundary marker
    from datetime import datetime as _dt
    logger.info("=" * 80)
    logger.info("NEW SESSION STARTED at %s", _dt.utcnow().isoformat())
    logger.info("=" * 80)
    return logger


def sanitize_filename(s: Any) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(s))


def candidate_column(columns: Sequence[str], candidates: Sequence[str], required: bool = False) -> Optional[str]:
    columns_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        if cand.lower() in columns_lower:
            return columns_lower[cand.lower()]
    if required:
        raise KeyError(f"Could not resolve required column from candidates={candidates}")
    return None


# --------------------------------------------
# NIfTI / image helpers
# --------------------------------------------

def load_nifti_array(path: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    path = str(path)
    if not path or not Path(path).exists():
        raise FileNotFoundError(path)

    try:
        import nibabel as nib  # type: ignore
        img = nib.load(path)
        arr = np.asarray(img.get_fdata(), dtype=np.float32)
        zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
        return arr, zooms
    except Exception:
        pass

    try:
        import SimpleITK as sitk  # type: ignore
        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        # FIX I-25: SITK GetArrayFromImage returns (z, y, x) — DO NOT transpose.
        # The previous np.transpose(arr, (2, 1, 0)) was WRONG: it produced (x, y, z)
        # which breaks find_best_slice() and central_masked_slice() that assume axis-0 = z (slices).
        # Also reorder spacing from SITK (x, y, z) to match array axes (z, y, x):
        sp = img.GetSpacing()  # (x_spacing, y_spacing, z_spacing)
        spacing = (float(sp[2]), float(sp[1]), float(sp[0]))  # -> (z, y, x)
        return arr, spacing
    except Exception as e:
        raise RuntimeError(f"Could not load NIfTI '{path}'. Install nibabel or SimpleITK. Original error: {e}")


def robust_zscore(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    scale = 1.4826 * mad if mad > 0 else float(np.std(values) + 1e-8)
    return (values - med) / (scale + 1e-8)


def central_masked_slice(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return np.zeros((16, 16), dtype=np.float32)
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)
    # choose axis with largest extent and take central slice along it
    extents = np.array([zmax - zmin + 1, ymax - ymin + 1, xmax - xmin + 1])
    axis = int(np.argmax(extents))
    center = int(np.round(coords[:, axis].mean()))
    if axis == 0:
        sl_img = image[center, :, :]
        sl_mask = mask[center, :, :]
    elif axis == 1:
        sl_img = image[:, center, :]
        sl_mask = mask[:, center, :]
    else:
        sl_img = image[:, :, center]
        sl_mask = mask[:, :, center]
    vals = sl_img[sl_mask > 0]
    if vals.size == 0:
        vals = sl_img.reshape(-1)
    vals = robust_zscore(vals)
    # compact embedding from masked central slice
    out = np.array([
        float(np.mean(vals)),
        float(np.std(vals)),
        float(np.percentile(vals, 10)),
        float(np.percentile(vals, 50)),
        float(np.percentile(vals, 90)),
    ], dtype=np.float32)
    return out


def gradient_energy(image: np.ndarray, mask: np.ndarray) -> float:
    if np.count_nonzero(mask) == 0:
        return 0.0
    grads = np.gradient(image.astype(np.float32))
    mag = np.sqrt(sum(g * g for g in grads))
    vals = mag[mask > 0]
    return float(np.mean(vals)) if vals.size else 0.0


def bbox_fill_ratio(mask: np.ndarray) -> float:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return 0.0
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    dims = (maxs - mins + 1).astype(float)
    bbox_vol = float(np.prod(dims)) if np.all(dims > 0) else 0.0
    return float(coords.shape[0] / bbox_vol) if bbox_vol > 0 else 0.0


def extract_single_lesion_image_features(image_path: str, mask_path: str, hist_bins: int = IMAGE_HIST_BINS) -> Dict[str, float]:
    image, spacing = load_nifti_array(image_path)
    mask, _ = load_nifti_array(mask_path)
    mask = (mask > 0).astype(np.uint8)

    if image.shape != mask.shape:
        raise ValueError(f"Image/mask shape mismatch: {image.shape} vs {mask.shape}")
    if np.count_nonzero(mask) == 0:
        raise ValueError("Mask is empty")

    voxel_volume_cc = float(np.prod(spacing) / 1000.0)
    lesion_vals = image[mask > 0].astype(np.float32)
    lesion_vals_z = robust_zscore(lesion_vals)
    hist_range = (-3.0, 3.0)
    hist, _ = np.histogram(np.clip(lesion_vals_z, *hist_range), bins=hist_bins, range=hist_range, density=True)
    hist = hist.astype(np.float32)

    features: Dict[str, float] = {
        "img_voxel_count": float(lesion_vals.size),
        "img_volume_cc": float(lesion_vals.size * voxel_volume_cc),
        "img_intensity_mean": float(np.mean(lesion_vals)),
        "img_intensity_std": float(np.std(lesion_vals)),
        "img_intensity_min": float(np.min(lesion_vals)),
        "img_intensity_max": float(np.max(lesion_vals)),
        "img_intensity_p10": float(np.percentile(lesion_vals, 10)),
        "img_intensity_p25": float(np.percentile(lesion_vals, 25)),
        "img_intensity_p50": float(np.percentile(lesion_vals, 50)),
        "img_intensity_p75": float(np.percentile(lesion_vals, 75)),
        "img_intensity_p90": float(np.percentile(lesion_vals, 90)),
        "img_intensity_iqr": float(np.percentile(lesion_vals, 75) - np.percentile(lesion_vals, 25)),
        "img_zscore_mean": float(np.mean(lesion_vals_z)),
        "img_zscore_std": float(np.std(lesion_vals_z)),
        "img_gradient_energy": gradient_energy(image, mask),
        "img_bbox_fill_ratio": bbox_fill_ratio(mask),
        "img_foreground_fraction": float(np.count_nonzero(mask) / mask.size),
        "img_spacing_x": float(spacing[0]),
        "img_spacing_y": float(spacing[1]),
        "img_spacing_z": float(spacing[2]),
    }

    center_embed = central_masked_slice(image, mask)
    for i, value in enumerate(center_embed.tolist(), start=1):
        features[f"img_central_slice_stat_{i}"] = float(value)

    for i, value in enumerate(hist.tolist(), start=1):
        features[f"img_hist_bin_{i:02d}"] = float(value)

    return features


# --------------------------------------------
# Data assembly
# --------------------------------------------

def resolve_input_paths() -> Tuple[Path, Path]:
    if not INPUT_STEP03_LESION.exists():
        raise FileNotFoundError(f"Missing Step 03 lesion table: {INPUT_STEP03_LESION}")
    if not INPUT_STEP04_PATIENT.exists():
        raise FileNotFoundError(f"Missing Step 04 patient table: {INPUT_STEP04_PATIENT}")
    return INPUT_STEP03_LESION, INPUT_STEP04_PATIENT


def build_lesion_image_feature_table(lesion_df: pd.DataFrame, cfg: Step06Config, logger: logging.Logger) -> pd.DataFrame:
    # FIX I-03: Check NIfTI library availability once before processing all rows
    _nifti_ok = False
    try:
        import nibabel as _nib  # noqa: F401
        _nifti_ok = True
    except ImportError:
        pass
    if not _nifti_ok:
        try:
            import SimpleITK as _sitk  # noqa: F401
            _nifti_ok = True
        except ImportError:
            pass
    if not _nifti_ok:
        raise RuntimeError(
            "Step 06 image feature extraction requires nibabel or SimpleITK. "
            "Install with: pip install nibabel --break-system-packages"
        )
    logger.info("NIfTI library check passed — proceeding with image feature extraction.")

    # FIX I-03 / I-04: Prioritize absolute paths produced by Step 03 (image_abs_path, mask_abs_path)
    image_col = candidate_column(
        lesion_df.columns,
        ["image_abs_path", "image_path", "img_path", "image", "nifti_path"],
        required=True
    )
    mask_col = candidate_column(
        lesion_df.columns,
        ["mask_abs_path", "mask_path", "mask", "segmentation_path"],
        required=True
    )
    patient_col = candidate_column(lesion_df.columns, ["patient_base", "patient_id", "patient", "patient_key"], required=True)
    case_col = candidate_column(lesion_df.columns, ["case_id", "filenameprefix", "study_id", "case", "visit_id"])

    records: List[Dict[str, Any]] = []
    n_ok = 0
    n_fail = 0
    for idx, row in lesion_df.iterrows():
        image_path = str(row[image_col])
        mask_path = str(row[mask_col])
        try:
            feats = extract_single_lesion_image_features(image_path, mask_path, hist_bins=cfg.image_hist_bins)
            record = {
                "patient_base": row[patient_col],
                "image_path": image_path,
                "mask_path": mask_path,
                "lesion_row_index": int(idx),
                "case_id": row[case_col] if case_col else None,
                "img_extract_status": "ok",
                **feats,
            }
            n_ok += 1
        except Exception as e:
            record = {
                "patient_base": row[patient_col],
                "image_path": image_path,
                "mask_path": mask_path,
                "lesion_row_index": int(idx),
                "case_id": row[case_col] if case_col else None,
                "img_extract_status": f"failed: {e}",
            }
            n_fail += 1
        records.append(record)

    logger.info(f"Image feature extraction completed: ok={n_ok}, failed={n_fail}")
    return pd.DataFrame(records)


def aggregate_image_features_to_patient(lesion_feature_df: pd.DataFrame, patient_df: pd.DataFrame) -> pd.DataFrame:
    ok_df = lesion_feature_df[lesion_feature_df["img_extract_status"] == "ok"].copy()
    if ok_df.empty:
        raise ValueError("No lesion image features were successfully extracted.")

    numeric_cols = [c for c in ok_df.columns if c.startswith("img_") and pd.api.types.is_numeric_dtype(ok_df[c])]
    grouped = ok_df.groupby("patient_base")
    agg_frames = []
    # FIX I-12: Compute non-std stats first, then std with ddof=0 to avoid NaN for single-lesion patients
    for stat_name, func in {
        "mean": "mean",
        "max": "max",
        "min": "min",
        "median": "median",
    }.items():
        sub = grouped[numeric_cols].agg(func)
        sub.columns = [f"patient_{c}_{stat_name}" for c in sub.columns]
        agg_frames.append(sub)
    # Population std (ddof=0): returns 0 for single-sample groups instead of NaN
    std_sub = grouped[numeric_cols].std(ddof=0).fillna(0.0)
    std_sub.columns = [f"patient_{c}_std" for c in std_sub.columns]
    agg_frames.append(std_sub)

    patient_img = pd.concat(agg_frames, axis=1).reset_index()
    counts = grouped.size().rename("n_image_lesions_used").reset_index()
    patient_img = patient_img.merge(counts, on="patient_base", how="left")
    patient_img["n_image_lesions_used"] = patient_img["n_image_lesions_used"].fillna(0)

    merged = patient_df.merge(patient_img, on="patient_base", how="left")
    for c in merged.columns:
        if c.startswith("patient_img_") or c == "n_image_lesions_used":
            if pd.api.types.is_numeric_dtype(merged[c]):
                merged[c] = merged[c].replace([np.inf, -np.inf], np.nan)
    return merged


def resolve_feature_sets(patient_df: pd.DataFrame, feature_dict_path: Path) -> Dict[str, List[str]]:
    feature_dict = load_json(feature_dict_path) if feature_dict_path.exists() else {}
    scanner_covariates = [c for c in feature_dict.get("scanner_covariate_columns", []) if c in patient_df.columns]
    radiomics_only = [c for c in feature_dict.get("numeric_feature_columns_no_scanner_covariates", []) if c in patient_df.columns]
    if not radiomics_only:
        exclude = {"patient_base", "split", "target_er", "target_pr", "target_her2", "ER", "PR", "HER2"}
        radiomics_only = [
            c for c in patient_df.columns
            if c not in exclude
            and pd.api.types.is_numeric_dtype(patient_df[c])
            and not c.startswith("patient_img_")
            and "field_strength" not in c.lower()
            and "manufacturer" not in c.lower()
            and "scanner" not in c.lower()
        ]
    image_only = [c for c in patient_df.columns if c.startswith("patient_img_") or c == "n_image_lesions_used"]
    if not scanner_covariates:
        scanner_covariates = [
            c for c in patient_df.columns
            if pd.api.types.is_numeric_dtype(patient_df[c]) and any(k in c.lower() for k in ("field_strength", "manufacturer", "scanner"))
        ]
    return {
        "image_only": image_only,
        "radiomics_only": radiomics_only,
        "scanner_only": scanner_covariates,
        "image_plus_radiomics": sorted(set(image_only + radiomics_only)),
        "hybrid_all": sorted(set(image_only + radiomics_only + scanner_covariates)),
    }


# --------------------------------------------
# Modeling components
# --------------------------------------------

def validate_binary_target(y: pd.Series, target_col: str):
    uniq = sorted(pd.Series(y.dropna().astype(int)).unique().tolist())
    if len(uniq) < 2:
        raise ValueError(f"{target_col}: only one class present: {uniq}")


def split_data(data: pd.DataFrame, feature_cols: List[str], target_col: str, cfg: Step06Config):
    data = data[~data[target_col].isna()].copy()
    data[target_col] = data[target_col].astype(int)
    validate_binary_target(data[target_col], target_col)
    X = data[list(feature_cols)].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    y = data[target_col].copy()
    s = data["split"].copy()
    X_train, y_train = X[s == "train"], y[s == "train"]
    X_valid, y_valid = X[s == "valid"], y[s == "valid"]
    X_test, y_test = X[s == "test"], y[s == "test"]
    if X_train.shape[0] < cfg.min_required_train_rows or X_valid.shape[0] < cfg.min_required_valid_rows or X_test.shape[0] < cfg.min_required_test_rows:
        raise ValueError("Insufficient split sizes.")
    counts = y_train.value_counts().to_dict()
    if min(counts.get(0, 0), counts.get(1, 0)) < cfg.min_class_count_train:
        raise ValueError(f"Insufficient minority class in train split: {counts}")
    return X_train, y_train, X_valid, y_valid, X_test, y_test


class SafeVarianceFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 1e-8):
        self.threshold = threshold
        self.support_mask_ = None

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        var = np.nanvar(X_arr, axis=0)
        self.support_mask_ = np.isfinite(var) & (var > self.threshold)
        if not np.any(self.support_mask_):
            self.support_mask_ = np.ones(X_arr.shape[1], dtype=bool)
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        return X_arr[:, self.support_mask_]

    def get_support(self):
        return self.support_mask_


class TopKByTrainAUCSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k: int = 20, min_features: int = 5, random_state: int = GLOBAL_SEED):
        self.k = k
        self.min_features = min_features
        self.random_state = random_state
        self.support_mask_ = None
        self.scores_ = None

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("Selector requires y.")
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y).astype(int)
        scores = []
        for j in range(X_arr.shape[1]):
            col = X_arr[:, j]
            if np.all(~np.isfinite(col)) or np.nanstd(col) == 0:
                scores.append(0.0)
                continue
            fill = np.nanmedian(col[np.isfinite(col)]) if np.any(np.isfinite(col)) else 0.0
            col = np.where(np.isfinite(col), col, fill)
            try:
                auc = roc_auc_score(y_arr, col)
                auc = max(auc, 1.0 - auc)  # orientation-free
                scores.append(float(auc))
            except Exception:
                scores.append(0.5)
        self.scores_ = np.asarray(scores, dtype=float)
        k_eff = min(max(self.min_features, self.k), X_arr.shape[1])
        order = np.argsort(self.scores_)[::-1]
        self.support_mask_ = np.zeros(X_arr.shape[1], dtype=bool)
        self.support_mask_[order[:k_eff]] = True
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        return X_arr[:, self.support_mask_]

    def get_support(self):
        return self.support_mask_


class OptionalPCAReducer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components: int = IMAGE_PCA_COMPONENTS):
        self.n_components = n_components
        self.reducer_ = None
        self.output_dim_ = None

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        n_features = X_arr.shape[1]
        n_samples = X_arr.shape[0]
        n_comp = min(self.n_components, n_features, max(1, n_samples - 1))
        if n_features <= n_comp:
            self.reducer_ = None
            self.output_dim_ = n_features
            return self
        self.reducer_ = PCA(n_components=n_comp, random_state=GLOBAL_SEED)
        self.reducer_.fit(X_arr)
        self.output_dim_ = n_comp
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        if self.reducer_ is None:
            return X_arr
        return self.reducer_.transform(X_arr)



def build_models_for_modality(modality_variant: str) -> Dict[str, Pipeline]:
    image_like = modality_variant in {"image_only", "image_plus_radiomics", "hybrid_all"}
    k = 12 if modality_variant == "image_only" else 20
    pre: List[Tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", SafeVarianceFilter(threshold=1e-8)),
        ("selector", TopKByTrainAUCSelector(k=k, min_features=min(5, k))),
    ]
    if image_like:
        pre.append(("pca", OptionalPCAReducer(n_components=IMAGE_PCA_COMPONENTS)))

    models = {
        "logistic": Pipeline(pre + [("scaler", StandardScaler()), ("model", LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", max_iter=4000, random_state=GLOBAL_SEED))]),
        "elastic_net": Pipeline(pre + [("scaler", RobustScaler()), ("model", LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.5, class_weight="balanced", max_iter=5000, random_state=GLOBAL_SEED))]),
        "svm_linear": Pipeline(pre + [("scaler", StandardScaler()), ("model", CalibratedClassifierCV(estimator=SVC(kernel="linear", C=0.5, class_weight="balanced", probability=False, random_state=GLOBAL_SEED), method="sigmoid", cv=3))]),
        "svm_rbf": Pipeline(pre + [("scaler", StandardScaler()), ("model", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=GLOBAL_SEED))]),
    }
    return models


# --------------------------------------------
# Metrics / plots
# --------------------------------------------

def compute_metrics(y_true: pd.Series, probs: np.ndarray, preds: np.ndarray):
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    return {
        "AUROC": float(roc_auc_score(y_true, probs)),
        "AUPRC": float(average_precision_score(y_true, probs)),
        "F1": float(f1_score(y_true, preds, zero_division=0)),
        "Balanced_Acc": float(balanced_accuracy_score(y_true, preds)),
        "Precision": float(precision_score(y_true, preds, zero_division=0)),
        "Recall_Sensitivity": float(recall_score(y_true, preds, zero_division=0)),
        "Specificity": specificity,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def choose_threshold_on_valid(y_valid: pd.Series, valid_probs: np.ndarray, threshold_grid: Sequence[float]):
    best_thr = 0.5
    best_metrics = None
    best_key = None
    for thr in threshold_grid:
        preds = (valid_probs >= thr).astype(int)
        m = compute_metrics(y_valid, valid_probs, preds)
        key = (m["Balanced_Acc"], m["F1"], m["AUPRC"])
        if best_key is None or key > best_key:
            best_key = key
            best_thr = float(thr)
            best_metrics = m
    return best_thr, best_metrics


def extract_selected_feature_names(model: Pipeline, original_feature_names: Sequence[str]) -> List[str]:
    names = list(original_feature_names)
    for step_name in ["variance", "selector"]:
        step = model.named_steps.get(step_name)
        if step is not None and hasattr(step, "get_support"):
            mask = np.asarray(step.get_support()).astype(bool)
            if len(mask) < len(names):
                mask = np.pad(mask, (0, len(names) - len(mask)), constant_values=False)
            else:
                mask = mask[: len(names)]
            names = [n for n, keep in zip(names, mask) if keep]
    if "pca" in model.named_steps:
        pca_step = model.named_steps["pca"]
        if getattr(pca_step, "reducer_", None) is not None:
            return [f"pca_component_{i+1}" for i in range(int(pca_step.output_dim_))]
    return names


def compute_feature_importance(model: Pipeline, X_reference: pd.DataFrame, y_reference: pd.Series, selected_feature_names: Sequence[str], random_state: int):
    final_model = model.named_steps["model"]
    base_model = final_model.estimator if isinstance(final_model, CalibratedClassifierCV) else final_model
    if hasattr(base_model, "coef_"):
        coef = np.asarray(base_model.coef_)
        if coef.ndim == 2:
            coef = coef[0]
        coef = np.abs(coef.astype(float)).ravel()
        features = list(selected_feature_names) if len(coef) == len(selected_feature_names) else [f"component_{i+1}" for i in range(len(coef))]
        native_df = pd.DataFrame({"feature": features, "importance": coef, "importance_type": "abs_coefficient"}).sort_values("importance", ascending=False)
    else:
        native_df = pd.DataFrame(columns=["feature", "importance", "importance_type"])

    try:
        perm = permutation_importance(model, X_reference, y_reference, scoring="roc_auc", n_repeats=20, random_state=random_state, n_jobs=1)
        perm_df = pd.DataFrame({
            "feature": list(X_reference.columns),
            "importance": perm.importances_mean,
            "importance_std": perm.importances_std,
            "importance_type": "permutation_on_test_locked__roc_auc",
        }).sort_values("importance", ascending=False)
    except Exception:
        perm_df = pd.DataFrame(columns=["feature", "importance", "importance_std", "importance_type"])
    return native_df.reset_index(drop=True), perm_df.reset_index(drop=True)


def save_heatmap(df: pd.DataFrame, path: Path, title: str, value_col: str = "AUROC"):
    pivot = df.pivot(index="target_display", columns="modality_variant", values=value_col)
    plt.figure(figsize=(10, 4))
    arr = pivot.values.astype(float)
    plt.imshow(arr, aspect="auto")
    plt.xticks(range(pivot.shape[1]), list(pivot.columns), rotation=30, ha="right")
    plt.yticks(range(pivot.shape[0]), list(pivot.index))
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if np.isfinite(val):
                plt.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9)
    plt.colorbar(label=value_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_grouped_bars(df: pd.DataFrame, path: Path, title: str, value_col: str = "AUROC"):
    targets = list(df["target_display"].drop_duplicates())
    variants = list(df["modality_variant"].drop_duplicates())
    x = np.arange(len(targets))
    width = 0.8 / max(1, len(variants))
    plt.figure(figsize=(12, 5))
    for i, variant in enumerate(variants):
        sub = df[df["modality_variant"] == variant].set_index("target_display")
        vals = [float(sub.loc[t, value_col]) if t in sub.index else np.nan for t in targets]
        plt.bar(x + i * width - 0.4 + width / 2, vals, width=width, label=variant)
    plt.xticks(x, targets)
    plt.ylabel(value_col)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_gap_plot(df: pd.DataFrame, path: Path):
    order = df.sort_values("generalization_gap_AUROC", ascending=False).reset_index(drop=True)
    labels = [f"{r.target_display}|{r.modality_variant}|{r.model}" for _, r in order.iterrows()]
    vals = order["generalization_gap_AUROC"].to_numpy(float)
    plt.figure(figsize=(14, 5))
    plt.bar(np.arange(len(vals)), vals)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xticks(np.arange(len(vals)), labels, rotation=70, ha="right")
    plt.ylabel("Validation AUROC - Test AUROC")
    plt.title("Generalization gap across image-vs-tabular experiments")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_roc_plot(y_true, probs, path: Path, title: str):
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc = roc_auc_score(y_true, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUROC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_pr_plot(y_true, probs, path: Path, title: str):
    precision, recall, _ = precision_recall_curve(y_true, probs)
    ap = average_precision_score(y_true, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, linewidth=2, label=f"AUPRC = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


# --------------------------------------------
# Core run logic
# --------------------------------------------

def run_single_modality_target(
    df: pd.DataFrame,
    feature_cols: List[str],
    modality_variant: str,
    target_col: str,
    cfg: Step06Config,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame, List[Path]]:
    target_name = TARGET_DISPLAY_NAMES.get(target_col, target_col)
    X_train, y_train, X_valid, y_valid, X_test, y_test = split_data(df, feature_cols, target_col, cfg)
    models = build_models_for_modality(modality_variant)
    valid_rows = []
    model_cache: Dict[str, Dict[str, Any]] = {}
    artifacts: List[Path] = []

    for model_name, model in models.items():
        try:
            model.fit(X_train, y_train)
            valid_probs = model.predict_proba(X_valid)[:, 1]
        except Exception as e:
            logger.warning(f"Skipping {target_col}/{modality_variant}/{model_name}: {e}")
            continue
        selected_features = extract_selected_feature_names(model, feature_cols)
        valid_metrics = compute_metrics(y_valid, valid_probs, (valid_probs >= 0.5).astype(int))
        valid_rows.append({
            "target": target_col,
            "target_display": target_name,
            "modality_variant": modality_variant,
            "model": model_name,
            "model_display": MODEL_DISPLAY_NAMES.get(model_name, model_name),
            "n_train": int(X_train.shape[0]),
            "n_valid": int(X_valid.shape[0]),
            "n_test": int(X_test.shape[0]),
            "n_input_features": int(len(feature_cols)),
            "n_selected_features": int(len(selected_features)),
            **valid_metrics,
        })
        model_cache[model_name] = {
            "model": model,
            "selected_features": selected_features,
            "valid_probs": valid_probs,
        }

    if not valid_rows:
        raise ValueError(f"No valid models for {target_col}/{modality_variant}")

    valid_df = pd.DataFrame(valid_rows).sort_values(["AUROC", "Balanced_Acc", "AUPRC", "F1"], ascending=False).reset_index(drop=True)
    best_model_name = str(valid_df.iloc[0]["model"])
    best_model = model_cache[best_model_name]["model"]
    best_selected_features = model_cache[best_model_name]["selected_features"]
    best_valid_probs = model_cache[best_model_name]["valid_probs"]
    best_threshold, best_valid_threshold_metrics = choose_threshold_on_valid(y_valid, best_valid_probs, cfg.prediction_threshold_grid)

    test_probs = best_model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= best_threshold).astype(int)
    test_metrics = compute_metrics(y_test, test_probs, test_preds)
    test_row = {
        "target": target_col,
        "target_display": target_name,
        "modality_variant": modality_variant,
        "model": best_model_name,
        "model_display": MODEL_DISPLAY_NAMES.get(best_model_name, best_model_name),
        "selected_on": "valid_AUROC",
        "threshold_selected_on_valid": best_threshold,
        "n_train": int(X_train.shape[0]),
        "n_valid": int(X_valid.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_input_features": int(len(feature_cols)),
        "n_selected_features": int(len(best_selected_features)),
        "train_positive_rate": float(y_train.mean()),
        "valid_positive_rate": float(y_valid.mean()),
        "test_positive_rate": float(y_test.mean()),
        **test_metrics,
    }
    test_row["validation_AUROC_at_selection"] = float(valid_df.iloc[0]["AUROC"])
    test_row["generalization_gap_AUROC"] = float(test_row["validation_AUROC_at_selection"] - test_row["AUROC"])

    rng = np.random.default_rng(cfg.global_seed)
    boot = []
    y_arr = np.asarray(y_test)
    p_arr = np.asarray(test_probs)
    for _ in range(cfg.max_bootstrap):
        idx = rng.integers(0, len(y_arr), len(y_arr))
        y_b = y_arr[idx]
        p_b = p_arr[idx]
        if len(np.unique(y_b)) < 2:
            continue
        boot.append(roc_auc_score(y_b, p_b))
    if len(boot) >= 100:
        test_row["auroc_bootstrap_ci95_lower"] = float(np.percentile(boot, 2.5))
        test_row["auroc_bootstrap_ci95_upper"] = float(np.percentile(boot, 97.5))
        test_row["auroc_bootstrap_n_valid"] = int(len(boot))

    safe_target = sanitize_filename(target_col)
    safe_modality = sanitize_filename(modality_variant)
    safe_model = sanitize_filename(best_model_name)

    if cfg.save_plots:
        roc_path = STEP06_FIGURES_DIR / f"step06_{safe_target}_{safe_modality}_{safe_model}_test_roc.png"
        pr_path = STEP06_FIGURES_DIR / f"step06_{safe_target}_{safe_modality}_{safe_model}_test_pr.png"
        save_roc_plot(y_test, test_probs, roc_path, f"{target_name} | {modality_variant} | {MODEL_DISPLAY_NAMES.get(best_model_name, best_model_name)} ROC")
        save_pr_plot(y_test, test_probs, pr_path, f"{target_name} | {modality_variant} | {MODEL_DISPLAY_NAMES.get(best_model_name, best_model_name)} PR")
        artifacts.extend([roc_path, pr_path])

    native_importance_df, perm_importance_df = compute_feature_importance(best_model, X_test, y_test, best_selected_features, cfg.global_seed)
    native_csv = STEP06_TABLES_DIR / f"step06_{safe_target}_{safe_modality}_{safe_model}_native_importance.csv"
    perm_csv = STEP06_TABLES_DIR / f"step06_{safe_target}_{safe_modality}_{safe_model}_permutation_importance.csv"
    selected_csv = STEP06_TABLES_DIR / f"step06_{safe_target}_{safe_modality}_{safe_model}_selected_features.csv"
    valid_csv = STEP06_TABLES_DIR / f"step06_{safe_target}_{safe_modality}_validation_table.csv"
    test_csv = STEP06_TABLES_DIR / f"step06_{safe_target}_{safe_modality}_best_test.csv"
    native_importance_df.to_csv(native_csv, index=False, encoding="utf-8-sig")
    perm_importance_df.to_csv(perm_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame({"selected_feature": best_selected_features}).to_csv(selected_csv, index=False, encoding="utf-8-sig")
    valid_df.to_csv(valid_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame([test_row]).to_csv(test_csv, index=False, encoding="utf-8-sig")
    artifacts.extend([native_csv, perm_csv, selected_csv, valid_csv, test_csv])

    summary = {
        "target": target_col,
        "display_name": target_name,
        "modality_variant": modality_variant,
        "n_input_features": int(len(feature_cols)),
        "best_model_name": best_model_name,
        "best_threshold": best_threshold,
        "best_model_valid_threshold_metrics": best_valid_threshold_metrics,
        "best_model_test_metrics": test_row,
        "top_native_features": native_importance_df.head(20).to_dict(orient="records"),
        "top_permutation_features": perm_importance_df.head(20).to_dict(orient="records"),
    }
    return valid_df, summary, pd.DataFrame([test_row]), artifacts


# --------------------------------------------
# Main
# --------------------------------------------

def main():
    _ap = argparse.ArgumentParser(add_help=False)
    _ap.add_argument("--force",    action="store_true")
    _ap.add_argument("--no-cache", action="store_true")
    _flags, _ = _ap.parse_known_args()

    cfg = Step06Config()
    set_global_seed(cfg.global_seed)
    for p in [REPORTS_DIR, FIGURES_DIR, TABLES_DIR, METADATA_DIR, LOGS_DIR, CHECKPOINTS_DIR, STEP06_FIGURES_DIR, STEP06_TABLES_DIR]:
        ensure_dir(p)

    log_file = LOGS_DIR / f"{STEP_NAME}.log"
    logger = setup_logger(log_file)
    state = StateManager(CHECKPOINTS_DIR / "pipeline_state.json")

    # ── Cache check ───────────────────────────────────────────────────────────
    _cfg_hash = hashlib.md5(json.dumps({
        "targets": sorted(cfg.target_columns), "variants": sorted(cfg.modality_variants),
        "hist_bins": cfg.image_hist_bins, "pca_components": cfg.image_pca_components,
        "seed": cfg.global_seed,
    }, sort_keys=True).encode()).hexdigest()
    _manifest = PROJECT_ROOT / "cache" / "step06_cache_manifest.json"
    _inputs   = [INPUT_STEP03_LESION, INPUT_STEP04_PATIENT]
    _outputs  = [STEP06_TABLES_DIR / "step06_model_results.csv", METADATA_DIR / "step06_image_baseline_summary.json"]
    if not _flags.force and not _flags.no_cache and _is_step_cached(_manifest, _inputs, _cfg_hash, _outputs):
        logger.info("Cache valid — Step 06 outputs unchanged. Skipping (use --force to re-run).")
        return

    lesion_csv, patient_csv = resolve_input_paths()
    lesion_df = pd.read_csv(lesion_csv)
    patient_df = pd.read_csv(patient_csv)
    if "split" not in patient_df.columns:
        raise ValueError("Missing split column in patient-level table.")
    patient_base_col = candidate_column(patient_df.columns, ["patient_base"], required=True)
    if patient_base_col != "patient_base":
        patient_df = patient_df.rename(columns={patient_base_col: "patient_base"})

    logger.info("=" * 90)
    logger.info(f"Starting {STEP_NAME}")
    logger.info(f"Loaded lesion table shape: {lesion_df.shape}")
    logger.info(f"Loaded patient table shape: {patient_df.shape}")

    lesion_feature_df = build_lesion_image_feature_table(lesion_df, cfg, logger)
    lesion_feature_csv = STEP06_TABLES_DIR / "step06_lesion_image_feature_table.csv"
    lesion_feature_df.to_csv(lesion_feature_csv, index=False, encoding="utf-8-sig")

    patient_merged_df = aggregate_image_features_to_patient(lesion_feature_df, patient_df)
    patient_merged_csv = STEP06_TABLES_DIR / "step06_patient_level_image_augmented_table.csv"
    patient_merged_df.to_csv(patient_merged_csv, index=False, encoding="utf-8-sig")

    feature_sets = resolve_feature_sets(patient_merged_df, INPUT_FEATURE_DICT)
    logger.info(f"Resolved modality sizes: { {k: len(v) for k, v in feature_sets.items()} }")

    all_valid_results = []
    all_best_results = []
    run_summary: Dict[str, Any] = {
        "config": asdict(cfg),
        "input_shapes": {
            "lesion_rows": int(lesion_df.shape[0]),
            "patient_rows": int(patient_df.shape[0]),
            "patient_image_augmented_rows": int(patient_merged_df.shape[0]),
            "patient_image_augmented_cols": int(patient_merged_df.shape[1]),
        },
        "modality_feature_set_sizes": {k: int(len(v)) for k, v in feature_sets.items()},
        "targets": {},
        "method_notes": {
            "purpose": "Benchmark image-native features extracted directly from MRI+mask against radiomics/scanner tables and their fusion.",
            "image_feature_definition": "Intensity, histogram, gradient, geometry, spacing, and central-slice embedding statistics extracted from tumor-masked MRI volumes.",
            "selection_policy": "All preprocessing and selection are fit on TRAIN only inside sklearn Pipelines. Model family and threshold are chosen on VALID only. TEST remains locked for final evaluation.",
        },
    }
    artifacts: Dict[str, str] = {
        "step06_log_file": str(log_file),
        "step06_lesion_image_feature_table_csv": str(lesion_feature_csv),
        "step06_patient_level_image_augmented_table_csv": str(patient_merged_csv),
    }

    for target_col in cfg.target_columns:
        if target_col not in patient_merged_df.columns:
            continue
        run_summary["targets"][target_col] = {}
        for modality_variant in cfg.modality_variants:
            feature_cols = feature_sets.get(modality_variant, [])
            if not feature_cols:
                run_summary["targets"][target_col][modality_variant] = {"status": "skipped_no_features"}
                continue
            try:
                valid_df, summary, best_df, target_artifacts = run_single_modality_target(
                    patient_merged_df, feature_cols, modality_variant, target_col, cfg, logger
                )
            except Exception as e:
                logger.warning(f"Skipping {target_col}/{modality_variant}: {e}")
                run_summary["targets"][target_col][modality_variant] = {"status": f"failed: {e}"}
                continue
            all_valid_results.append(valid_df)
            all_best_results.append(best_df)
            run_summary["targets"][target_col][modality_variant] = summary
            for p in target_artifacts:
                artifacts[p.stem] = str(p)

    if not all_best_results:
        raise ValueError("No Step 06 final results were generated.")

    valid_master_df = pd.concat(all_valid_results, axis=0, ignore_index=True)
    best_master_df = pd.concat(all_best_results, axis=0, ignore_index=True)

    valid_master_csv = STEP06_TABLES_DIR / "step06_validation_model_comparison_master_table.csv"
    best_master_csv = STEP06_TABLES_DIR / "step06_model_results.csv"
    valid_master_df.to_csv(valid_master_csv, index=False, encoding="utf-8-sig")
    best_master_df.to_csv(best_master_csv, index=False, encoding="utf-8-sig")
    artifacts["step06_validation_model_comparison_master_table_csv"] = str(valid_master_csv)
    artifacts["step06_model_results_csv"] = str(best_master_csv)

    best_by_target = (
        best_master_df.sort_values(["target", "AUROC", "AUPRC", "Balanced_Acc", "F1"], ascending=[True, False, False, False, False])
        .groupby("target", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best_by_target_csv = STEP06_TABLES_DIR / "step06_best_models_by_target.csv"
    best_by_target.to_csv(best_by_target_csv, index=False, encoding="utf-8-sig")
    artifacts["step06_best_models_by_target_csv"] = str(best_by_target_csv)

    if cfg.save_plots:
        heatmap_path = STEP06_FIGURES_DIR / "step06_modality_heatmap_auroc.png"
        bars_path = STEP06_FIGURES_DIR / "step06_modality_grouped_bars_auroc.png"
        gap_path = STEP06_FIGURES_DIR / "step06_generalization_gap.png"
        save_heatmap(best_master_df, heatmap_path, "Image vs tabular modality comparison (test AUROC)")
        save_grouped_bars(best_master_df, bars_path, "Cross-modal benchmark by target", value_col="AUROC")
        save_gap_plot(best_master_df, gap_path)
        artifacts["step06_modality_heatmap_auroc_png"] = str(heatmap_path)
        artifacts["step06_modality_grouped_bars_auroc_png"] = str(bars_path)
        artifacts["step06_generalization_gap_png"] = str(gap_path)

    # Compare against Step 05 if available
    cross_stage_comparison = []
    if INPUT_STEP05_RESULTS.exists():
        step05_df = pd.read_csv(INPUT_STEP05_RESULTS)
        if "target" in step05_df.columns and "AUROC" in step05_df.columns:
            step05_best = step05_df.sort_values(["target", "AUROC"], ascending=[True, False]).groupby("target", as_index=False).head(1)
            step05_best = step05_best[["target", "AUROC", "AUPRC", "model_display", "feature_set_variant"]].rename(columns={
                "AUROC": "step05_best_auroc",
                "AUPRC": "step05_best_auprc",
                "model_display": "step05_best_model_display",
                "feature_set_variant": "step05_best_variant",
            })
            cross_stage_df = best_by_target.merge(step05_best, on="target", how="left")
            cross_stage_df["delta_auroc_step06_minus_step05"] = cross_stage_df["AUROC"] - cross_stage_df["step05_best_auroc"]
            cross_stage_csv = STEP06_TABLES_DIR / "step06_vs_step05_best_by_target.csv"
            cross_stage_df.to_csv(cross_stage_csv, index=False, encoding="utf-8-sig")
            artifacts["step06_vs_step05_best_by_target_csv"] = str(cross_stage_csv)
            cross_stage_comparison = cross_stage_df.to_dict(orient="records")

    summary_path = METADATA_DIR / "step06_image_baseline_summary.json"
    run_summary["overall_best_by_target"] = best_by_target.to_dict(orient="records")
    run_summary["cross_stage_comparison_to_step05"] = cross_stage_comparison
    run_summary["validation_master_csv"] = str(valid_master_csv)
    run_summary["final_test_results_csv"] = str(best_master_csv)
    save_json(run_summary, summary_path)
    artifacts["step06_image_baseline_summary_json"] = str(summary_path)

    if cfg.save_publication_package:
        zip_path = REPORTS_DIR / "step06_image_benchmark_publication_package.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(STEP06_TABLES_DIR.glob("step06_*")):
                zf.write(p, arcname=f"tables/{p.name}")
            for p in sorted(STEP06_FIGURES_DIR.glob("step06_*")):
                zf.write(p, arcname=f"figures/{p.name}")
            zf.write(summary_path, arcname=f"metadata/{summary_path.name}")
        artifacts["step06_image_benchmark_publication_package_zip"] = str(zip_path)

    state.mark_step_done(STEP_NAME, artifacts=artifacts)
    state.add_note("Step 06 image-based validation and cross-modal comparison completed successfully.")
    _save_step_manifest(_manifest, _inputs, _cfg_hash)
    logger.info("Completed Step 06 successfully.")
    logger.info("=" * 90)


if __name__ == "__main__":
    main()
