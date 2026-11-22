from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import warnings
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    lgb = None

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    xgb = None

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in divide",
    category=RuntimeWarning,
    module=r"sklearn\.feature_selection\._univariate_selection",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("./bcbm_project").resolve()
STEP_NAME = "step_05_tabular_modeling"
INPUT_PATIENT_LEVEL = PROJECT_ROOT / "data" / "processed" / "analysis_ready_step04_patient_level.csv"
INPUT_FEATURE_DICT = PROJECT_ROOT / "metadata" / "step04_feature_dictionary.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
METADATA_DIR = PROJECT_ROOT / "metadata"
LOGS_DIR = PROJECT_ROOT / "logs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# ── Hyperparameters ───────────────────────────────────────────────────────────
TARGET_COLUMNS = ["target_er", "target_pr", "target_her2"]
TARGET_DISPLAY_NAMES = {"target_er": "ER", "target_pr": "PR", "target_her2": "HER2"}
FEATURE_SET_VARIANTS = ["radiomics_pure", "radiomics_plus_burden", "all_features", "acquisition_only"]

MODEL_DISPLAY_NAMES = {
    "elastic_net": "Elastic Net (L1+L2)",
    "elastic_net_strong": "Elastic Net (Strong)",
    "elastic_net_sparse": "Elastic Net (Sparse)",
    "lgbm": "LightGBM",
    "lgbm_strong": "LightGBM (Strong Reg.)",
    "xgb": "XGBoost",
    "extra_trees": "Extra Trees",
    "svm_linear": "Linear SVM (Calibrated)",
    "pls_logistic": "PLS + Logistic",
}

GLOBAL_SEED = 42
OUTER_CV_SPLITS = 5
INNER_CV_SPLITS = 3
STABILITY_N_BOOTSTRAP = 50
STABILITY_MAX_FEATURES = 20          # tighter than 30 for n≈139
CORR_THRESHOLD = 0.88                # tighter redundancy control
MAX_MISSING_RATIO = 0.40             # remove unstable columns inside fold only
COMBAT_MIN_BATCH_SIZE = 3
PERMUTATION_REPEATS = 20
SAVE_PLOTS = True
SAVE_PUBLICATION_PACKAGE = True
MIN_REQUIRED_TRAINVAL_ROWS = 15
MIN_REQUIRED_TEST_ROWS = 4
MIN_CLASS_COUNT = 2
CALIBRATION_CV_SPLITS = 3

BATCH_COLUMN_CANDIDATES = [
    "dominant_manufacturer",
    "magnetic_field_strength_label",
    "dominant_field_strength_t",
]


@dataclass
class Step05Config:
    project_root: str = str(PROJECT_ROOT)
    step_name: str = STEP_NAME
    input_patient_level_csv: str = str(INPUT_PATIENT_LEVEL)
    input_feature_dict_json: str = str(INPUT_FEATURE_DICT)
    target_columns: List[str] = None
    feature_set_variants: List[str] = None
    global_seed: int = GLOBAL_SEED
    outer_cv_splits: int = OUTER_CV_SPLITS
    inner_cv_splits: int = INNER_CV_SPLITS
    stability_max_features: int = STABILITY_MAX_FEATURES
    stability_n_bootstrap: int = STABILITY_N_BOOTSTRAP
    corr_threshold: float = CORR_THRESHOLD
    max_missing_ratio: float = MAX_MISSING_RATIO
    permutation_repeats: int = PERMUTATION_REPEATS
    save_plots: bool = SAVE_PLOTS
    save_publication_package: bool = SAVE_PUBLICATION_PACKAGE
    min_required_trainval_rows: int = MIN_REQUIRED_TRAINVAL_ROWS
    min_required_test_rows: int = MIN_REQUIRED_TEST_ROWS
    min_class_count: int = MIN_CLASS_COUNT
    calibration_cv_splits: int = CALIBRATION_CV_SPLITS

    def __post_init__(self):
        if self.target_columns is None:
            self.target_columns = TARGET_COLUMNS.copy()
        if self.feature_set_variants is None:
            self.feature_set_variants = FEATURE_SET_VARIANTS.copy()


class StateManager:
    def __init__(self, state_path: Path):
        self.state_path = state_path
        self.state = self._load()

    def _load(self):
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


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_json(data, path: Path):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sanitize_filename(s):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(s))


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
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    sh = logging.StreamHandler()
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    from datetime import datetime as _dt
    logger.info("=" * 80)
    logger.info("NEW SESSION STARTED at %s", _dt.utcnow().isoformat())
    logger.info("=" * 80)
    return logger


def detect_runtime_environment() -> Dict[str, Any]:
    import platform
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "xgboost_version": getattr(xgb, "__version__", None) if XGB_AVAILABLE else None,
        "lightgbm_version": getattr(lgb, "__version__", None) if LGB_AVAILABLE else None,
        "gpu_available": False,
        "gpu_name": None,
    }
    try:
        import torch
        info["gpu_available"] = bool(torch.cuda.is_available())
        info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as e:
        info["torch_probe_error"] = str(e)
    return info


def resolve_feature_sets(df: pd.DataFrame, feature_dict_path: Path) -> Dict[str, List[str]]:
    fdict = load_json(feature_dict_path) if feature_dict_path.exists() else {}
    scanner_covariates = [c for c in fdict.get("scanner_covariate_columns", []) if c in df.columns]
    all_numeric = [c for c in fdict.get("numeric_feature_columns", []) if c in df.columns]
    rad_only = [c for c in fdict.get("numeric_feature_columns_no_scanner_covariates", []) if c in df.columns]
    if not all_numeric:
        exclude = {"patient_base", "split", "stratum_key", "ER", "PR", "HER2", "ER_bin", "PR_bin", "HER2_bin", "target_er", "target_pr", "target_her2"}
        all_numeric = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if not rad_only:
        rad_only = [c for c in all_numeric if c not in set(scanner_covariates)]
    scanner_only = [c for c in scanner_covariates if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not scanner_only:
        kw = ("field_strength", "manufacturer", "scanner")
        scanner_only = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and any(k in c.lower() for k in kw)]
    # Prefer the explicit blocks written by step 04. Excluding the eight named scanner covariates
    # is not enough to make a set "radiomics only": the patient-level table repeats field strength
    # as mean/min/max/std_field_strength_t and patient_meta_magnetic_field_strength_id_*, and also
    # carries pixel spacing and slice geometry. Those belong with acquisition, or the radiomics arm
    # competes against its own control. Blocks: radiomic (642) / burden-study (75) / acquisition (25).
    rad_block = [c for c in fdict.get("radiomic_block_columns", []) if c in df.columns]
    acq_block = [c for c in fdict.get("acquisition_block_columns", []) if c in df.columns]
    bur_block = [c for c in fdict.get("burden_study_block_columns", []) if c in df.columns]
    if rad_block and acq_block:
        return {
            "radiomics_pure": list(rad_block),
            "radiomics_plus_burden": rad_block + bur_block,
            "all_features": rad_block + bur_block + acq_block,
            "acquisition_only": list(acq_block),
            "_radiomics_only_cols": list(rad_block),
        }
    return {
        "radiomics_plus_scanner": list(all_numeric),
        "radiomics_only": list(rad_only),
        "scanner_only": list(scanner_only),
        "_radiomics_only_cols": list(rad_only),
    }


def get_batch_labels(df: pd.DataFrame) -> Optional[np.ndarray]:
    for col in BATCH_COLUMN_CANDIDATES:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.notna().mean() > 0.5 and vals.nunique(dropna=True) >= 2:
                return vals.fillna(-1).astype(int).values
    return None


def validate_binary_target(y: pd.Series, col: str):
    uniq = sorted(y.dropna().astype(int).unique().tolist())
    if len(uniq) < 2:
        raise ValueError(f"{col}: only one class present: {uniq}")


class SimpleCombatHarmonizer:
    """
    Simplified ComBat-style scanner harmonization.
    Fit only on the current training fold. This version is NaN-safe: all-NaN columns
    are mapped to mean=0/std=1 so downstream fold-local preprocessing can handle them.
    """
    def __init__(self, min_batch_size: int = COMBAT_MIN_BATCH_SIZE):
        self.min_batch_size = min_batch_size
        self._fitted = False

    def fit(self, X: np.ndarray, batch: np.ndarray) -> "SimpleCombatHarmonizer":
        X = np.asarray(X, dtype=float)
        batch = np.asarray(batch).ravel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            grand_mean = np.nanmean(X, axis=0)
            grand_std = np.nanstd(X, axis=0)
        self.grand_mean_ = np.where(np.isnan(grand_mean), 0.0, grand_mean)
        self.grand_std_ = np.where(np.isnan(grand_std) | (grand_std < 1e-8), 1.0, grand_std)
        self.batch_stats_: Dict[Any, Dict[str, np.ndarray]] = {}
        for b in np.unique(batch):
            m = batch == b
            if int(m.sum()) < self.min_batch_size:
                self.batch_stats_[b] = {"mean": self.grand_mean_.copy(), "std": self.grand_std_.copy()}
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    bm = np.nanmean(X[m], axis=0)
                    bs = np.nanstd(X[m], axis=0)
                bm = np.where(np.isnan(bm), self.grand_mean_, bm)
                bs = np.where(np.isnan(bs) | (bs < 1e-8), 1.0, bs)
                self.batch_stats_[b] = {"mean": bm, "std": bs}
        self._fitted = True
        return self

    def transform(self, X: np.ndarray, batch: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        X = np.asarray(X, dtype=float).copy()
        batch = np.asarray(batch).ravel()
        for b in np.unique(batch):
            m = batch == b
            if not m.any():
                continue
            stats = self.batch_stats_.get(b, {"mean": self.grand_mean_, "std": self.grand_std_})
            X[m] = (X[m] - stats["mean"]) / stats["std"] * self.grand_std_ + self.grand_mean_
        return X


def apply_combat_to_fold(
    X_tr: pd.DataFrame,
    X_val: pd.DataFrame,
    batch_tr: np.ndarray,
    batch_val: np.ndarray,
    harm_cols: List[str],
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not harm_cols or len(np.unique(batch_tr)) < 2:
        return X_tr, X_val
    cols_present = [c for c in harm_cols if c in X_tr.columns]
    if not cols_present:
        return X_tr, X_val
    try:
        harmonizer = SimpleCombatHarmonizer()
        X_tr_arr = X_tr[cols_present].values.astype(float)
        X_val_arr = X_val[cols_present].values.astype(float)
        harmonizer.fit(X_tr_arr, batch_tr)
        X_tr_out = X_tr.copy()
        X_val_out = X_val.copy()
        X_tr_out[cols_present] = harmonizer.transform(X_tr_arr, batch_tr)
        X_val_out[cols_present] = harmonizer.transform(X_val_arr, batch_val)
        return X_tr_out, X_val_out
    except Exception as e:
        logger.warning("ComBat failed: %s — using raw features for this fold.", e)
        return X_tr, X_val


class MissingnessFilter(BaseEstimator, TransformerMixin):
    """Fold-local unsupervised filter; removes columns that are mostly/all missing in training fold."""
    def __init__(self, max_missing_ratio: float = 0.40, min_features: int = 5):
        self.max_missing_ratio = max_missing_ratio
        self.min_features = min_features

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        self.n_features_in_ = X_arr.shape[1]
        miss = np.mean(np.isnan(X_arr), axis=0)
        keep = np.where(miss <= self.max_missing_ratio)[0].tolist()
        if len(keep) < min(self.min_features, self.n_features_in_):
            order = np.argsort(miss)
            keep = order[: min(self.min_features, self.n_features_in_)].tolist()
        self.keep_indices_ = sorted(set(int(i) for i in keep))
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        return X_arr[:, self.keep_indices_]

    def get_support(self) -> np.ndarray:
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.keep_indices_] = True
        return mask


class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.88, min_features: int = 5):
        self.threshold = threshold
        self.min_features = min_features

    def fit(self, X, y=None):
        df = pd.DataFrame(np.asarray(X, dtype=float))
        self.n_features_in_ = df.shape[1]
        if df.shape[1] <= self.min_features:
            self.keep_indices_ = list(range(df.shape[1]))
            return self
        corr = df.corr().abs().fillna(0.0)
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = set(i for i, col in enumerate(upper.columns) if (upper[col] > self.threshold).any())
        keep = [i for i in range(df.shape[1]) if i not in to_drop]
        if len(keep) < min(self.min_features, df.shape[1]):
            keep = list(range(min(self.min_features, df.shape[1])))
        self.keep_indices_ = keep
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_indices_]

    def get_support(self) -> np.ndarray:
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.keep_indices_] = True
        return mask


class StabilitySelector(BaseEstimator, TransformerMixin):
    def __init__(self, max_features: int = 20, n_bootstrap: int = 50, sample_fraction: float = 0.8, selection_threshold: float = 0.45, random_state: int = 42):
        self.max_features = max_features
        self.n_bootstrap = n_bootstrap
        self.sample_fraction = sample_fraction
        self.selection_threshold = selection_threshold
        self.random_state = random_state

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("StabilitySelector requires y.")
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y).astype(int)
        n_samples, n_features = X_arr.shape
        if n_features == 0:
            raise ValueError("No features available for StabilitySelector.")
        sample_size = max(int(n_samples * self.sample_fraction), MIN_CLASS_COUNT * 2 + 1)
        sample_size = min(sample_size, n_samples)
        rng = np.random.default_rng(self.random_state)
        feature_counts = np.zeros(n_features, dtype=float)
        k_per_boot = max(1, min(self.max_features, n_features) // 2)
        n_valid = 0
        for _ in range(self.n_bootstrap):
            idx = rng.choice(n_samples, size=sample_size, replace=False)
            Xb, yb = X_arr[idx], y_arr[idx]
            if len(np.unique(yb)) < 2:
                continue
            f_scores, _ = _safe_f_classif(Xb, yb)
            try:
                mi_scores = mutual_info_classif(Xb, yb, random_state=self.random_state, discrete_features=False)
                mi_scores = np.nan_to_num(mi_scores, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                mi_scores = np.zeros_like(f_scores)
            score = _rank_normalize(f_scores) + _rank_normalize(mi_scores)
            top = np.argsort(score)[::-1][:k_per_boot]
            feature_counts[top] += 1
            n_valid += 1
        if n_valid == 0:
            self.stability_scores_ = np.ones(n_features) / n_features
            mask = np.zeros(n_features, dtype=bool)
            mask[: min(5, n_features)] = True
            self.support_mask_ = mask
            return self
        self.stability_scores_ = feature_counts / n_valid
        mask = self.stability_scores_ >= self.selection_threshold
        min_keep = min(5, n_features)
        if mask.sum() < min_keep:
            order = np.argsort(self.stability_scores_)[::-1]
            mask = np.zeros(n_features, dtype=bool)
            mask[order[:min_keep]] = True
        if mask.sum() > self.max_features:
            order = np.argsort(self.stability_scores_)[::-1]
            mask = np.zeros(n_features, dtype=bool)
            mask[order[: self.max_features]] = True
        self.support_mask_ = mask
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.support_mask_]

    def get_support(self) -> np.ndarray:
        return self.support_mask_


class PLSBinaryClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components: int = 3, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y).astype(int)
        n_comp = min(self.n_components, max(1, X_arr.shape[1]), max(1, X_arr.shape[0] - 1))
        self.pls_ = PLSRegression(n_components=n_comp)
        Z = self.pls_.fit_transform(X_arr, y_arr)[0]
        self.lr_ = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=self.random_state)
        self.lr_.fit(Z, y_arr)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        Z = self.pls_.transform(np.asarray(X, dtype=float))
        return self.lr_.predict_proba(Z)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def coef_(self):
        return self.lr_.coef_


def _safe_f_classif(X: np.ndarray, y: np.ndarray):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores, pvalues = f_classif(X, y)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    pvalues = np.nan_to_num(np.asarray(pvalues, dtype=float), nan=1.0, posinf=1.0, neginf=1.0)
    return scores, pvalues


def _rank_normalize(scores: np.ndarray) -> np.ndarray:
    scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if np.all(scores == scores[0]):
        return np.zeros_like(scores)
    ranks = pd.Series(scores).rank(method="average").to_numpy(dtype=float)
    return (ranks - ranks.min()) / max(ranks.max() - ranks.min(), 1e-12)


def youden_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    y_arr = np.asarray(y_true).astype(int)
    p_arr = np.asarray(probs, dtype=float)
    if len(np.unique(y_arr)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_arr, p_arr)
    j = tpr - fpr
    best = int(np.argmax(j))
    thr = float(thresholds[best])
    if not np.isfinite(thr):
        thr = 0.5
    return float(np.clip(thr, 0.10, 0.90))


def compute_metrics(y_true: pd.Series, probs: np.ndarray, preds: np.ndarray) -> Dict[str, Any]:
    y_arr = np.asarray(y_true).astype(int)
    p_arr = np.asarray(probs, dtype=float)
    pred_arr = np.asarray(preds).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_arr, pred_arr, labels=[0, 1]).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    auroc = float(roc_auc_score(y_arr, p_arr)) if len(np.unique(y_arr)) > 1 else float("nan")
    auprc = float(average_precision_score(y_arr, p_arr)) if len(np.unique(y_arr)) > 1 else float("nan")
    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "F1": float(f1_score(y_arr, pred_arr, zero_division=0)),
        "Balanced_Acc": float(balanced_accuracy_score(y_arr, pred_arr)),
        "Precision": float(precision_score(y_arr, pred_arr, zero_division=0)),
        "Recall_Sensitivity": float(recall_score(y_arr, pred_arr, zero_division=0)),
        "Specificity": spec,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def build_preprocessing_steps(cfg: Step05Config) -> list:
    return [
        ("missingness", MissingnessFilter(max_missing_ratio=cfg.max_missing_ratio, min_features=5)),
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("variance", VarianceThreshold(threshold=1e-8)),
        ("correlation", CorrelationFilter(threshold=cfg.corr_threshold, min_features=5)),
        ("stability", StabilitySelector(
            max_features=cfg.stability_max_features,
            n_bootstrap=cfg.stability_n_bootstrap,
            sample_fraction=0.8,
            selection_threshold=0.45,
            random_state=cfg.global_seed,
        )),
    ]


def calibration_cv_from_y(y: pd.Series, desired: int) -> int:
    counts = y.value_counts()
    return max(2, min(desired, int(counts.min()) if not counts.empty else 2))


def build_candidate_models(cfg: Step05Config, y_train: pd.Series, gpu: bool) -> Dict[str, Pipeline]:
    pre = build_preprocessing_steps(cfg)
    cal_cv = calibration_cv_from_y(y_train, cfg.calibration_cv_splits)
    models: Dict[str, Pipeline] = {}

    models["elastic_net"] = Pipeline(pre + [("scaler", RobustScaler()), ("model", LogisticRegression(
        penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.5,
        class_weight="balanced", max_iter=6000, random_state=cfg.global_seed,
    ))])
    models["elastic_net_strong"] = Pipeline(pre + [("scaler", RobustScaler()), ("model", LogisticRegression(
        penalty="elasticnet", solver="saga", l1_ratio=0.8, C=0.1,
        class_weight="balanced", max_iter=6000, random_state=cfg.global_seed,
    ))])
    models["elastic_net_sparse"] = Pipeline(pre + [("scaler", RobustScaler()), ("model", LogisticRegression(
        penalty="elasticnet", solver="saga", l1_ratio=0.95, C=0.05,
        class_weight="balanced", max_iter=6000, random_state=cfg.global_seed,
    ))])

    if LGB_AVAILABLE:
        models["lgbm"] = Pipeline(pre + [("model", lgb.LGBMClassifier(
            n_estimators=80, max_depth=2, learning_rate=0.04,
            num_leaves=7, min_child_samples=12, subsample=0.75, colsample_bytree=0.65,
            reg_alpha=2.0, reg_lambda=4.0, class_weight="balanced",
            random_state=cfg.global_seed, n_jobs=-1, verbose=-1,
        ))])
        models["lgbm_strong"] = Pipeline(pre + [("model", lgb.LGBMClassifier(
            n_estimators=50, max_depth=2, learning_rate=0.03,
            num_leaves=5, min_child_samples=15, subsample=0.70, colsample_bytree=0.55,
            reg_alpha=5.0, reg_lambda=8.0, class_weight="balanced",
            random_state=cfg.global_seed, n_jobs=-1, verbose=-1,
        ))])

    if XGB_AVAILABLE:
        models["xgb"] = Pipeline(pre + [("model", xgb.XGBClassifier(
            n_estimators=60, max_depth=2, learning_rate=0.04,
            subsample=0.75, colsample_bytree=0.55, min_child_weight=6,
            reg_lambda=5.0, reg_alpha=2.0, eval_metric="logloss",
            random_state=cfg.global_seed, tree_method="hist", device="cuda" if gpu else "cpu",
        ))])

    models["extra_trees"] = Pipeline(pre + [("model", ExtraTreesClassifier(
        n_estimators=200, max_depth=3, min_samples_leaf=5,
        max_features="sqrt", class_weight="balanced",
        random_state=cfg.global_seed, n_jobs=-1,
    ))])

    svc = SVC(kernel="linear", C=0.15, class_weight="balanced", probability=False, random_state=cfg.global_seed)
    models["svm_linear"] = Pipeline(pre + [("scaler", StandardScaler()), ("model", CalibratedClassifierCV(estimator=svc, method="sigmoid", cv=cal_cv))])
    models["pls_logistic"] = Pipeline(pre + [("scaler", RobustScaler()), ("model", PLSBinaryClassifier(n_components=3, random_state=cfg.global_seed))])
    return models


def extract_selected_feature_names(model: Pipeline, original_names: List[str]) -> List[str]:
    names = list(original_names)
    for step_key in ["missingness", "variance", "correlation", "stability", "selector"]:
        step = model.named_steps.get(step_key)
        if step is None or not hasattr(step, "get_support"):
            continue
        mask = np.asarray(step.get_support()).astype(bool)
        if len(mask) < len(names):
            mask = np.pad(mask, (0, len(names) - len(mask)), constant_values=False)
        else:
            mask = mask[: len(names)]
        names = [n for n, keep in zip(names, mask) if keep]
    return names


def compute_feature_importance(model: Pipeline, X_ref: pd.DataFrame, y_ref: pd.Series, selected_names: List[str], seed: int, n_repeats: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    final = model.named_steps["model"]
    base = final.estimator if isinstance(final, CalibratedClassifierCV) else final
    if hasattr(base, "feature_importances_"):
        vals = np.asarray(base.feature_importances_, dtype=float).ravel()
        feats = list(selected_names) if len(vals) == len(selected_names) else [f"latent_{i+1}" for i in range(len(vals))]
        native_df = pd.DataFrame({"feature": feats, "importance": vals, "importance_type": "model_native"}).sort_values("importance", ascending=False)
    elif hasattr(base, "coef_"):
        coef = np.asarray(base.coef_)
        if coef.ndim == 2:
            coef = coef[0]
        coef = np.abs(coef.astype(float)).ravel()
        feats = list(selected_names) if len(coef) == len(selected_names) else [f"latent_{i+1}" for i in range(len(coef))]
        native_df = pd.DataFrame({"feature": feats, "importance": coef, "importance_type": "abs_coefficient"}).sort_values("importance", ascending=False)
    else:
        native_df = pd.DataFrame(columns=["feature", "importance", "importance_type"])
    try:
        perm = permutation_importance(model, X_ref, y_ref, scoring="roc_auc", n_repeats=n_repeats, random_state=seed, n_jobs=1)
        perm_df = pd.DataFrame({
            "feature": list(X_ref.columns),
            "importance": perm.importances_mean,
            "importance_std": perm.importances_std,
            "importance_type": "permutation_roc_auc_locked_test",
        }).sort_values("importance", ascending=False)
    except Exception:
        perm_df = pd.DataFrame(columns=["feature", "importance", "importance_std", "importance_type"])
    return native_df.reset_index(drop=True), perm_df.reset_index(drop=True)


def save_roc_plot(y_true, probs, path: Path, title: str):
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc = roc_auc_score(y_true, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUROC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(title)
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()


def save_pr_plot(y_true, probs, path: Path, title: str):
    precision, recall, _ = precision_recall_curve(y_true, probs)
    ap = average_precision_score(y_true, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, lw=2, label=f"AUPRC = {ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(title)
    plt.legend(loc="lower left"); plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()


def save_nested_cv_plot(fold_results: List[Dict], path: Path, title: str):
    folds = [r["fold"] + 1 for r in fold_results]
    aurocs = [r["outer_auroc"] if not np.isnan(r["outer_auroc"]) else 0 for r in fold_results]
    mean_a = np.nanmean([r["outer_auroc"] for r in fold_results])
    plt.figure(figsize=(7, 4))
    plt.bar(folds, aurocs, alpha=0.75)
    plt.axhline(mean_a, linestyle="--", label=f"Mean = {mean_a:.3f}")
    plt.ylim(0, 1); plt.xlabel("Outer Fold"); plt.ylabel("AUROC")
    plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()


def save_model_comparison_plot(results_df: pd.DataFrame, path: Path, title: str):
    order_col = "nested_cv_auroc_mean" if "nested_cv_auroc_mean" in results_df.columns else "AUROC"
    order = results_df.sort_values(order_col, ascending=False)
    labels = [f"{TARGET_DISPLAY_NAMES.get(t, t)} | {v} | {MODEL_DISPLAY_NAMES.get(m, m)}" for t, v, m in zip(order["target"], order["feature_set_variant"], order["model"])]
    vals = order[order_col].to_numpy(dtype=float)
    plt.figure(figsize=(max(14, len(vals) * 1.2), 7))
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), labels, rotation=65, ha="right")
    plt.ylabel(order_col); plt.title(title); plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()


def harmonize_pair_if_needed(X_tr, X_val, batch_tr, batch_val, harm_cols, use_combat, logger):
    if use_combat:
        return apply_combat_to_fold(X_tr, X_val, batch_tr, batch_val, harm_cols, logger)
    return X_tr, X_val


def run_nested_cv_pipeline(
    df: pd.DataFrame,
    feature_cols: List[str],
    harm_cols: List[str],
    batch_labels: Optional[np.ndarray],
    feature_set_variant: str,
    target_col: str,
    cfg: Step05Config,
    figures_dir: Path,
    tables_dir: Path,
    logger: logging.Logger,
    runtime_info: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict, List[Path], pd.DataFrame]:
    target_name = TARGET_DISPLAY_NAMES.get(target_col, target_col)
    gpu = bool(runtime_info.get("gpu_available", False))

    data = df[~df[target_col].isna()].copy()
    data[target_col] = data[target_col].astype(int)
    validate_binary_target(data[target_col], target_col)

    X_cols = list(feature_cols)
    for c in X_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data[X_cols] = data[X_cols].replace([np.inf, -np.inf], np.nan)

    mask_tv = data["split"].isin(["train", "valid"])
    mask_test = data["split"] == "test"
    if mask_tv.sum() < cfg.min_required_trainval_rows:
        raise ValueError(f"Insufficient train+valid rows: {mask_tv.sum()}")
    if mask_test.sum() < cfg.min_required_test_rows:
        raise ValueError(f"Insufficient test rows: {mask_test.sum()}")

    X_tv = data.loc[mask_tv, X_cols].copy()
    y_tv = data.loc[mask_tv, target_col].copy()
    X_test = data.loc[mask_test, X_cols].copy()
    y_test = data.loc[mask_test, target_col].copy()

    use_combat = batch_labels is not None and feature_set_variant != "acquisition_only" and len(harm_cols) > 0
    if use_combat:
        bl_aligned = pd.Series(batch_labels, index=df.index).reindex(data.index).fillna(-1).astype(int)
        batch_tv = bl_aligned.loc[mask_tv].values
        batch_test = bl_aligned.loc[mask_test].values
    else:
        batch_tv = np.zeros(len(y_tv), dtype=int)
        batch_test = np.zeros(len(y_test), dtype=int)

    n_outer = min(cfg.outer_cv_splits, int(y_tv.value_counts().min()))
    n_outer = max(n_outer, 2)
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=cfg.global_seed)
    outer_fold_results: List[Dict] = []
    oof_probs = np.full(len(y_tv), np.nan)
    logger.info("  [%s | %s] Outer CV splits=%d", target_col, feature_set_variant, n_outer)

    for fold_i, (tr_idx, val_idx) in enumerate(outer_cv.split(X_tv, y_tv)):
        X_otr_raw = X_tv.iloc[tr_idx].copy()
        y_otr = y_tv.iloc[tr_idx].copy()
        X_oval_raw = X_tv.iloc[val_idx].copy()
        y_oval = y_tv.iloc[val_idx].copy()

        n_inner = min(cfg.inner_cv_splits, int(y_otr.value_counts().min()), len(y_otr))
        n_inner = max(n_inner, 2)
        inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=cfg.global_seed + fold_i)
        candidates = build_candidate_models(cfg, y_otr, gpu)
        inner_scores: Dict[str, float] = {}

        for mn, model in candidates.items():
            fold_aucs = []
            for in_tr_i, in_val_i in inner_cv.split(X_otr_raw, y_otr):
                X_itr_raw = X_otr_raw.iloc[in_tr_i].copy()
                y_itr = y_otr.iloc[in_tr_i].copy()
                X_ival_raw = X_otr_raw.iloc[in_val_i].copy()
                y_ival = y_otr.iloc[in_val_i].copy()
                if len(np.unique(y_itr)) < 2 or len(np.unique(y_ival)) < 2:
                    continue
                X_itr, X_ival = harmonize_pair_if_needed(
                    X_itr_raw,
                    X_ival_raw,
                    batch_tv[tr_idx][in_tr_i],
                    batch_tv[tr_idx][in_val_i],
                    harm_cols,
                    use_combat,
                    logger,
                )
                try:
                    model.fit(X_itr, y_itr)
                    ip = model.predict_proba(X_ival)[:, 1]
                    fold_aucs.append(float(roc_auc_score(y_ival, ip)))
                except Exception as e:
                    logger.debug("  Inner %s fold failed: %s", mn, e)
            inner_scores[mn] = float(np.mean(fold_aucs)) if fold_aucs else 0.0

        best_inner = max(inner_scores, key=inner_scores.get) if inner_scores else "elastic_net"
        X_otr, X_oval = harmonize_pair_if_needed(X_otr_raw, X_oval_raw, batch_tv[tr_idx], batch_tv[val_idx], harm_cols, use_combat, logger)
        best_model_outer = build_candidate_models(cfg, y_otr, gpu)[best_inner]
        try:
            best_model_outer.fit(X_otr, y_otr)
            outer_probs = best_model_outer.predict_proba(X_oval)[:, 1]
        except Exception as e:
            logger.warning("  Outer fold %d fit failed (%s): %s", fold_i, best_inner, e)
            outer_probs = None

        if outer_probs is not None:
            oof_probs[val_idx] = outer_probs
            fold_auc = float(roc_auc_score(y_oval, outer_probs)) if len(np.unique(y_oval)) > 1 else float("nan")
        else:
            # Leave oof_probs[val_idx] as NaN (its initialized value) so this failed
            # fold is excluded from Youden threshold selection (valid_oof mask) and
            # from the nested AUROC mean (np.nanmean), instead of injecting synthetic
            # 0.5 predictions that would bias both.
            fold_auc = float("nan")
        outer_fold_results.append({
            "fold": fold_i,
            "best_model_inner": best_inner,
            "inner_scores": inner_scores,
            "outer_auroc": fold_auc,
            "n_train": int(len(tr_idx)),
            "n_val": int(len(val_idx)),
        })
        logger.info("    Fold %d/%d | winner=%s inner=%.3f outer_AUROC=%.3f", fold_i + 1, n_outer, best_inner, inner_scores.get(best_inner, 0), fold_auc)

    nested_auroc_mean = float(np.nanmean([r["outer_auroc"] for r in outer_fold_results]))
    nested_auroc_std = float(np.nanstd([r["outer_auroc"] for r in outer_fold_results]))
    model_votes = Counter([r["best_model_inner"] for r in outer_fold_results])
    max_votes = max(model_votes.values())
    tied = [m for m, v in model_votes.items() if v == max_votes]
    if len(tied) > 1:
        final_model_name = max(tied, key=lambda m: float(np.nanmean([r["inner_scores"].get(m, 0.0) for r in outer_fold_results])))
    else:
        final_model_name = model_votes.most_common(1)[0][0]
    logger.info("  Final model: %s  votes=%s  Nested AUROC=%.3f±%.3f", final_model_name, dict(model_votes), nested_auroc_mean, nested_auroc_std)

    valid_oof = ~np.isnan(oof_probs)
    best_threshold = youden_threshold(y_tv.values[valid_oof], oof_probs[valid_oof]) if valid_oof.sum() >= 10 and len(np.unique(y_tv.values[valid_oof])) > 1 else 0.5
    logger.info("  Youden threshold (OOF): %.3f", best_threshold)

    X_tv_harm, X_test_harm = harmonize_pair_if_needed(X_tv, X_test, batch_tv, batch_test, harm_cols, use_combat, logger)
    final_model = build_candidate_models(cfg, y_tv, gpu)[final_model_name]
    final_model.fit(X_tv_harm, y_tv)
    selected_features = extract_selected_feature_names(final_model, X_cols)

    test_probs = final_model.predict_proba(X_test_harm)[:, 1]
    test_preds = (test_probs >= best_threshold).astype(int)
    test_metrics = compute_metrics(y_test, test_probs, test_preds)

    boot_rows: List[float] = []
    try:
        rng = np.random.default_rng(cfg.global_seed)
        ya, pa = np.asarray(y_test), np.asarray(test_probs)
        for _ in range(1000):
            idx = rng.integers(0, len(ya), len(ya))
            yb, pb = ya[idx], pa[idx]
            if len(np.unique(yb)) > 1:
                boot_rows.append(float(roc_auc_score(yb, pb)))
    except Exception:
        pass

    all_model_names = list(outer_fold_results[0]["inner_scores"].keys()) if outer_fold_results else list(build_candidate_models(cfg, y_tv, gpu).keys())
    valid_rows = []
    for mn in all_model_names:
        fold_aucs = [r["inner_scores"].get(mn, float("nan")) for r in outer_fold_results]
        valid_rows.append({
            "target": target_col,
            "target_display": target_name,
            "scenario": "nested_cv_leakage_safe",
            "feature_set_variant": feature_set_variant,
            "model": mn,
            "model_display": MODEL_DISPLAY_NAMES.get(mn, mn),
            "mean_inner_cv_auroc": float(np.nanmean(fold_aucs)),
            "std_inner_cv_auroc": float(np.nanstd(fold_aucs)),
            "n_outer_folds_as_winner": int(model_votes.get(mn, 0)),
        })
    valid_df = pd.DataFrame(valid_rows).sort_values("mean_inner_cv_auroc", ascending=False).reset_index(drop=True)

    best_test_row: Dict[str, Any] = {
        "target": target_col,
        "target_display": target_name,
        "scenario": "nested_cv_leakage_safe",
        "feature_set_variant": feature_set_variant,
        "model": final_model_name,
        "model_display": MODEL_DISPLAY_NAMES.get(final_model_name, final_model_name),
        "selection_method": "nested_cv_inner_auroc_majority_vote_tiebreak_mean_inner",
        "threshold_method": "youden_index_oof",
        "threshold": float(best_threshold),
        "n_trainval": int(X_tv.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_input_features": int(len(feature_cols)),
        "n_selected_features": int(len(selected_features)),
        "nested_cv_auroc_mean": nested_auroc_mean,
        "nested_cv_auroc_std": nested_auroc_std,
        "nested_cv_n_outer_folds": int(n_outer),
        "combat_harmonization_applied": bool(use_combat),
        "stability_selection_max_features": int(cfg.stability_max_features),
        "max_missing_ratio": float(cfg.max_missing_ratio),
        "train_positive_rate": float(y_tv.mean()),
        "test_positive_rate": float(y_test.mean()),
        **test_metrics,
    }
    best_test_row["validation_AUROC_at_selection"] = nested_auroc_mean
    best_test_row["generalization_gap_AUROC"] = round(nested_auroc_mean - test_metrics["AUROC"], 4)
    best_test_row["generalization_gap"] = best_test_row["generalization_gap_AUROC"]
    if len(boot_rows) >= 100:
        best_test_row["auroc_bootstrap_ci95_lower"] = float(np.percentile(boot_rows, 2.5))
        best_test_row["auroc_bootstrap_ci95_upper"] = float(np.percentile(boot_rows, 97.5))
        best_test_row["auroc_bootstrap_n"] = int(len(boot_rows))

    best_test_df = pd.DataFrame([best_test_row])
    native_imp, perm_imp = compute_feature_importance(final_model, X_test_harm, y_test, selected_features, cfg.global_seed, cfg.permutation_repeats)

    st = sanitize_filename(target_col)
    sv = sanitize_filename(feature_set_variant)
    sm = sanitize_filename(final_model_name)
    valid_csv = tables_dir / f"step05master_{st}_{sv}_inner_cv_comparison.csv"
    test_csv = tables_dir / f"step05master_{st}_{sv}_best_model_test_result.csv"
    native_csv = tables_dir / f"step05master_{st}_{sv}_{sm}_feature_importance_native.csv"
    perm_csv = tables_dir / f"step05master_{st}_{sv}_{sm}_feature_importance_permutation.csv"
    selected_csv = tables_dir / f"step05master_{st}_{sv}_{sm}_selected_features.csv"
    valid_df.to_csv(valid_csv, index=False, encoding="utf-8-sig")
    best_test_df.to_csv(test_csv, index=False, encoding="utf-8-sig")
    native_imp.to_csv(native_csv, index=False, encoding="utf-8-sig")
    perm_imp.to_csv(perm_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame({"selected_feature": selected_features}).to_csv(selected_csv, index=False, encoding="utf-8-sig")
    artifacts: List[Path] = [valid_csv, test_csv, native_csv, perm_csv, selected_csv]

    if cfg.save_plots:
        roc_p = figures_dir / f"step05master_{st}_{sv}_{sm}_test_roc.png"
        pr_p = figures_dir / f"step05master_{st}_{sv}_{sm}_test_pr.png"
        ncv_p = figures_dir / f"step05master_{st}_{sv}_nested_cv_folds.png"
        title_base = f"{target_name} | {feature_set_variant} | {MODEL_DISPLAY_NAMES.get(final_model_name, final_model_name)}"
        save_roc_plot(y_test, test_probs, roc_p, f"{title_base} ROC")
        save_pr_plot(y_test, test_probs, pr_p, f"{title_base} PR")
        save_nested_cv_plot(outer_fold_results, ncv_p, f"{target_name} | {feature_set_variant} Nested CV")
        artifacts.extend([roc_p, pr_p, ncv_p])

    target_summary = {
        "target": target_col,
        "display_name": target_name,
        "feature_set_variant": feature_set_variant,
        "n_input_features": int(len(feature_cols)),
        "final_model_name": final_model_name,
        "threshold": float(best_threshold),
        "nested_cv_auroc_mean": nested_auroc_mean,
        "nested_cv_auroc_std": nested_auroc_std,
        "combat_applied": bool(use_combat),
        "best_model_test_metrics": best_test_row,
        "outer_fold_results": outer_fold_results,
        "top_native_features": native_imp.head(20).to_dict(orient="records"),
        "top_permutation_features": perm_imp.head(20).to_dict(orient="records"),
    }
    return valid_df, target_summary, artifacts, best_test_df


def main():
    _ap = argparse.ArgumentParser(add_help=False)
    _ap.add_argument("--force",    action="store_true")
    _ap.add_argument("--no-cache", action="store_true")
    _flags, _ = _ap.parse_known_args()

    cfg = Step05Config()
    set_global_seed(cfg.global_seed)
    for p in [REPORTS_DIR, FIGURES_DIR, TABLES_DIR, METADATA_DIR, LOGS_DIR, CHECKPOINTS_DIR]:
        ensure_dir(p)

    log_file = LOGS_DIR / f"{STEP_NAME}.log"
    logger = setup_logger(log_file)
    state = StateManager(CHECKPOINTS_DIR / "pipeline_state.json")

    # ── Cache check ───────────────────────────────────────────────────────────
    _cfg_hash = hashlib.md5(json.dumps({
        "outer": cfg.outer_cv_splits, "inner": cfg.inner_cv_splits,
        "max_feat": cfg.stability_max_features, "n_boot": cfg.stability_n_bootstrap,
        "corr": cfg.corr_threshold, "seed": cfg.global_seed,
        "targets": sorted(cfg.target_columns), "variants": sorted(cfg.feature_set_variants),
    }, sort_keys=True).encode()).hexdigest()
    _manifest = PROJECT_ROOT / "cache" / "tabular_preprocessing" / "step05_cache_manifest.json"
    _inputs   = [INPUT_PATIENT_LEVEL, INPUT_FEATURE_DICT]
    _outputs  = [TABLES_DIR / "step05master_model_results.csv", METADATA_DIR / "step05master_summary.json"]
    if not _flags.force and not _flags.no_cache and _is_step_cached(_manifest, _inputs, _cfg_hash, _outputs):
        logger.info("Cache valid — Step 05 outputs unchanged. Skipping (use --force to re-run).")
        return

    if not INPUT_PATIENT_LEVEL.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_PATIENT_LEVEL}")
    df = pd.read_csv(INPUT_PATIENT_LEVEL)
    if "split" not in df.columns:
        raise ValueError("Missing 'split' column.")

    feature_sets = resolve_feature_sets(df, INPUT_FEATURE_DICT)
    harm_cols = feature_sets.pop("_radiomics_only_cols")
    runtime_info = detect_runtime_environment()
    batch_labels = get_batch_labels(df)

    logger.info("Starting %s", STEP_NAME)
    logger.info("Input shape: %s", df.shape)
    logger.info("Feature set sizes: %s", {k: len(v) for k, v in feature_sets.items()})
    logger.info("LightGBM available: %s | XGBoost available: %s", LGB_AVAILABLE, XGB_AVAILABLE)
    logger.info("Batch labels found: %s  unique batches: %s", batch_labels is not None, len(np.unique(batch_labels)) if batch_labels is not None else 0)
    logger.info("max_features=%d | max_missing=%.2f | corr=%.2f | Outer CV=%d | Inner CV=%d", cfg.stability_max_features, cfg.max_missing_ratio, cfg.corr_threshold, cfg.outer_cv_splits, cfg.inner_cv_splits)

    all_valid: List[pd.DataFrame] = []
    all_test: List[pd.DataFrame] = []
    run_summary: Dict[str, Any] = {
        "config": asdict(cfg),
        "runtime": runtime_info,
        "input_shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "feature_set_sizes": {k: len(v) for k, v in feature_sets.items()},
        "split_counts": df["split"].value_counts(dropna=False).to_dict(),
        "combat_available": batch_labels is not None,
        "method_notes": {
            "approach": "nested_cross_validation",
            "outer_splits": cfg.outer_cv_splits,
            "inner_splits": cfg.inner_cv_splits,
            "harmonization": "SimpleCombat fit separately inside inner folds, outer folds, and final trainval-only fit",
            "feature_selection": f"missingness + variance + correlation + stability_selection (max={cfg.stability_max_features})",
            "threshold_selection": "youden_index_from_oof_probabilities",
            "leakage_guard": "ComBat, feature filtering, feature selection, scaling, and models are fit only on training partitions; test remains locked until final evaluation.",
            "interpretation_warning": "Prefer nested_cv_auroc_mean±std over locked-test AUROC when the test set is very small.",
        },
        "targets": {},
    }
    artifacts: Dict[str, str] = {"log_file": str(log_file)}

    for target_col in cfg.target_columns:
        if target_col not in df.columns:
            continue
        run_summary["targets"][target_col] = {}
        for variant in cfg.feature_set_variants:
            feature_cols = feature_sets.get(variant, [])
            if not feature_cols:
                run_summary["targets"][target_col][variant] = {"status": "skipped_no_features"}
                continue
            logger.info("=" * 60)
            logger.info("TARGET=%s  VARIANT=%s", target_col, variant)
            try:
                valid_df, target_summary, t_artifacts, best_test_df = run_nested_cv_pipeline(
                    df=df,
                    feature_cols=feature_cols,
                    harm_cols=harm_cols,
                    batch_labels=batch_labels,
                    feature_set_variant=variant,
                    target_col=target_col,
                    cfg=cfg,
                    figures_dir=FIGURES_DIR,
                    tables_dir=TABLES_DIR,
                    logger=logger,
                    runtime_info=runtime_info,
                )
            except Exception as e:
                logger.warning("FAILED %s/%s: %s", target_col, variant, e)
                run_summary["targets"][target_col][variant] = {"status": f"failed: {e}"}
                continue
            all_valid.append(valid_df)
            all_test.append(best_test_df)
            run_summary["targets"][target_col][variant] = target_summary
            for p in t_artifacts:
                artifacts[p.stem] = str(p)
            logger.info(
                "  DONE: nested_AUROC=%.3f±%.3f  test_AUROC=%.3f  gap=%.3f  threshold=%.3f  features=%d",
                target_summary["nested_cv_auroc_mean"],
                target_summary["nested_cv_auroc_std"],
                target_summary["best_model_test_metrics"]["AUROC"],
                target_summary["best_model_test_metrics"]["generalization_gap_AUROC"],
                target_summary["threshold"],
                target_summary["best_model_test_metrics"]["n_selected_features"],
            )

    if not all_test:
        raise ValueError("No results generated.")

    inner_cv_master = pd.concat(all_valid, ignore_index=True).sort_values(["target", "feature_set_variant", "mean_inner_cv_auroc"], ascending=[True, True, False])
    inner_cv_csv = TABLES_DIR / "step05master_inner_cv_comparison_master.csv"
    inner_cv_master.to_csv(inner_cv_csv, index=False, encoding="utf-8-sig")
    artifacts["inner_cv_master_csv"] = str(inner_cv_csv)

    # Backward-compatible alias for Step 10 or older analysis scripts.
    validation_alias_csv = TABLES_DIR / "step05master_validation_model_comparison_master_table.csv"
    inner_cv_master.to_csv(validation_alias_csv, index=False, encoding="utf-8-sig")
    artifacts["step05master_validation_model_comparison_master_table_csv"] = str(validation_alias_csv)

    test_master = pd.concat(all_test, ignore_index=True).sort_values(["target", "feature_set_variant"], ascending=True)
    results_csv = TABLES_DIR / "step05master_model_results.csv"
    test_master.to_csv(results_csv, index=False, encoding="utf-8-sig")
    artifacts["model_results_csv"] = str(results_csv)
    artifacts["step05master_model_results_csv"] = str(results_csv)

    # IMPORTANT: choose global best by nested CV, not by small locked-test AUROC.
    global_best = (
        test_master.sort_values(["target", "nested_cv_auroc_mean", "nested_cv_auroc_std", "AUROC"], ascending=[True, False, True, False])
        .groupby("target", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    global_best_csv = TABLES_DIR / "step05master_global_best_models_by_target.csv"
    global_best.to_csv(global_best_csv, index=False, encoding="utf-8-sig")
    artifacts["global_best_csv"] = str(global_best_csv)
    artifacts["step05master_global_best_models_by_target_csv"] = str(global_best_csv)

    variant_summary = (
        test_master.groupby(["target", "feature_set_variant"], as_index=False)[["AUROC", "AUPRC", "Balanced_Acc", "F1", "nested_cv_auroc_mean", "generalization_gap_AUROC"]]
        .max()
        .sort_values(["target", "nested_cv_auroc_mean"], ascending=[True, False])
    )
    variant_csv = TABLES_DIR / "step05master_feature_set_variant_summary.csv"
    variant_summary.to_csv(variant_csv, index=False, encoding="utf-8-sig")
    artifacts["variant_summary_csv"] = str(variant_csv)
    artifacts["step05master_feature_set_variant_summary_csv"] = str(variant_csv)

    scenario_summary = (
        test_master.groupby(["target", "scenario"], as_index=False)[["AUROC", "AUPRC", "Balanced_Acc", "F1", "nested_cv_auroc_mean"]]
        .max()
        .sort_values(["target", "nested_cv_auroc_mean"], ascending=[True, False])
    )
    scenario_csv = TABLES_DIR / "step05master_scenario_summary.csv"
    scenario_summary.to_csv(scenario_csv, index=False, encoding="utf-8-sig")
    artifacts["step05master_scenario_summary_csv"] = str(scenario_csv)

    if cfg.save_plots:
        cmp_path = FIGURES_DIR / "step05master_model_comparison_auroc.png"
        save_model_comparison_plot(test_master, cmp_path, "Nested CV | Feature-set × Target — AUROC")
        artifacts["comparison_plot"] = str(cmp_path)
        artifacts["step05master_model_comparison_auroc_png"] = str(cmp_path)

    run_summary["overall_best_by_target"] = global_best.to_dict(orient="records")
    run_summary["variant_summary"] = variant_summary.to_dict(orient="records")
    run_summary["final_test_results_csv"] = str(results_csv)
    run_summary["validation_master_csv"] = str(validation_alias_csv)
    run_summary["variant_summary_csv"] = str(variant_csv)
    run_summary["scenario_summary_csv"] = str(scenario_csv)

    summary_path = METADATA_DIR / "step05master_summary.json"
    save_json(run_summary, summary_path)
    artifacts["summary_json"] = str(summary_path)
    artifacts["step05master_summary_json"] = str(summary_path)

    if cfg.save_publication_package:
        zip_path = REPORTS_DIR / "step05master_publication_package.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(TABLES_DIR.glob("step05master_*")):
                zf.write(p, arcname=f"tables/{p.name}")
            for p in sorted(FIGURES_DIR.glob("step05master_*")):
                zf.write(p, arcname=f"figures/{p.name}")
            zf.write(summary_path, arcname=f"metadata/{summary_path.name}")
        artifacts["publication_zip"] = str(zip_path)
        artifacts["step05master_publication_package_zip"] = str(zip_path)

    state.mark_step_done(STEP_NAME, artifacts=artifacts)
    state.add_note("Step 05 improved leakage-safe nested CV pipeline completed.")
    _save_step_manifest(_manifest, _inputs, _cfg_hash)
    plt.close("all")
    logger.info("Completed %s successfully.", STEP_NAME)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
