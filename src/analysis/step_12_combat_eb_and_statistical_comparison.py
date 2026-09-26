from __future__ import annotations

"""
Step 12 — EB-ComBat Sensitivity + Statistical Model Comparison
==============================================================

Purpose
-------
This script adds two publication-level sensitivity analyses to the BCBM project
without modifying Steps 00–11:

1) EB-ComBat sensitivity analysis
   - Re-runs Step 05-like tabular modeling on BCBM patient-level data.
   - Compares:
       a) no ComBat / raw radiomics
       b) Step05 SimpleCombat-style fold-safe harmonization
       c) EB-style ComBat harmonization with empirical shrinkage
   - Harmonization is always fit on train/trainval only and then applied to
     validation/test, preserving leakage safety.

2) Statistical comparison of paired AUROC values
   - Generates locked-test predictions for the selected models/variants.
   - Computes pairwise AUROC differences using:
       a) DeLong test, when mathematically valid
       b) paired bootstrap CI and p-value, preferred for small cohorts
   - Also compares Step 11 external proxy predictions when available.

File location
-------------
    src/analysis/step_12_combat_eb_and_statistical_comparison.py

Run after Step 05 and Step 11:
    python src/analysis/step_12_combat_eb_and_statistical_comparison.py

Main outputs
------------
    bcbm_project/reports/tables/step12_ebcombat_sensitivity_results.csv
    bcbm_project/reports/tables/step12_locked_test_predictions.csv
    bcbm_project/reports/tables/step12_internal_pairwise_model_comparison.csv
    bcbm_project/reports/tables/step12_external_pairwise_model_comparison.csv
    bcbm_project/metadata/step12_summary.json
    bcbm_project/reports/figures/step12_*.png

Notes
-----
- This script intentionally uses the radiomic block (radiomics_pure) as the primary sensitivity
  setting because scanner covariates do not have direct one-to-one equivalents
  in OpenBTAI and scanner-only models are not biologically interpretable for
  external proxy validation.
- EB-style ComBat here uses empirical shrinkage of batch means/variances toward
  the global distribution. If neuroCombat is installed, you can extend this
  script to use it, but this implementation is dependency-light and leakage-safe.
"""

import argparse
import hashlib
import importlib.util
import json
import logging
import math
import os
import random
import sys
import warnings
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
# Do not hide pandas FutureWarnings globally; fix dtype assignment explicitly below.


# -----------------------------------------------------------------------------
# Paths / config
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path("./bcbm_project").resolve()
STEP_NAME = "step_12_combat_eb_and_statistical_comparison"

INPUT_PATIENT_LEVEL = PROJECT_ROOT / "data" / "processed" / "analysis_ready_step04_patient_level.csv"
INPUT_FEATURE_DICT = PROJECT_ROOT / "metadata" / "step04_feature_dictionary.json"
STEP05_RESULTS = PROJECT_ROOT / "reports" / "tables" / "step05master_model_results.csv"
STEP05_GLOBAL_BEST = PROJECT_ROOT / "reports" / "tables" / "step05master_global_best_models_by_target.csv"
STEP11_PREDICTIONS = PROJECT_ROOT / "reports" / "tables" / "step11_openbtai_external_proxy_predictions.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"
METADATA_DIR = PROJECT_ROOT / "metadata"
LOGS_DIR = PROJECT_ROOT / "logs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

TARGET_COLUMNS = ["target_er", "target_pr", "target_her2"]
TARGET_DISPLAY = {"target_er": "ER", "target_pr": "PR", "target_her2": "HER2"}
DEFAULT_FEATURE_VARIANTS = ["radiomics_pure", "all_features"]
DEFAULT_COMBAT_MODES = ["none", "simple_combat", "eb_combat"]
GLOBAL_SEED = 42
BOOTSTRAP_N = 3000
BOOTSTRAP_ALPHA = 0.05
OUTER_CV_SPLITS = 5
INNER_CV_SPLITS = 3

BATCH_COLUMN_CANDIDATES = [
    "dominant_manufacturer",
    "magnetic_field_strength_label",
    "dominant_field_strength_t",
]


@dataclass
class Step12Config:
    project_root: str = str(PROJECT_ROOT)
    step_name: str = STEP_NAME
    input_patient_level_csv: str = str(INPUT_PATIENT_LEVEL)
    input_feature_dict_json: str = str(INPUT_FEATURE_DICT)
    step05_results_csv: str = str(STEP05_RESULTS)
    step05_global_best_csv: str = str(STEP05_GLOBAL_BEST)
    step11_predictions_csv: str = str(STEP11_PREDICTIONS)
    target_columns: List[str] = None
    feature_variants: List[str] = None
    combat_modes: List[str] = None
    global_seed: int = GLOBAL_SEED
    outer_cv_splits: int = OUTER_CV_SPLITS
    inner_cv_splits: int = INNER_CV_SPLITS
    bootstrap_n: int = BOOTSTRAP_N
    bootstrap_alpha: float = BOOTSTRAP_ALPHA
    eb_prior_strength: float = 8.0
    run_ebcombat_sensitivity: bool = True
    run_internal_pairwise_comparison: bool = True
    run_external_pairwise_comparison: bool = True
    save_plots: bool = True
    save_zip: bool = True

    def __post_init__(self):
        if self.target_columns is None:
            self.target_columns = TARGET_COLUMNS.copy()
        if self.feature_variants is None:
            self.feature_variants = DEFAULT_FEATURE_VARIANTS.copy()
        if self.combat_modes is None:
            self.combat_modes = DEFAULT_COMBAT_MODES.copy()


# -----------------------------------------------------------------------------
# Logging / IO
# -----------------------------------------------------------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger(STEP_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    ensure_dir(log_file.parent)
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


def save_json(data: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def import_step05_module():
    """
    Import Step 05 from correct project structure.

    Supports:
    - src/modeling/step_05_tabular_modeling.py (current structure)
    - root fallback (legacy)
    """

    candidates = [
        # ✅ structure الحالي
        Path.cwd() / "src" / "modeling" / "step_05_tabular_modeling.py",

        # fallback
        Path.cwd() / "step_05_tabular_modeling.py",

        # في حال التشغيل من داخل src/analysis
        Path(__file__).resolve().parents[2] / "src" / "modeling" / "step_05_tabular_modeling.py",

        # fallback إضافي
        Path(__file__).resolve().parents[2] / "step_05_tabular_modeling.py",
    ]

    path = next((p.resolve() for p in candidates if p.exists()), None)

    if path is None:
        searched = "\n".join(str(p.resolve()) for p in candidates)
        raise FileNotFoundError(
            "Cannot find step_05_tabular_modeling.py. Searched:\n"
            f"{searched}\n\n"
            "Fix: ensure Step 05 exists in src/modeling/"
        )

    module_name = "step05_for_step12"

    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load import spec for {path}")

    mod = importlib.util.module_from_spec(spec)

    # مهم للـ dataclass داخل Step 05
    sys.modules[module_name] = mod

    spec.loader.exec_module(mod)

    return mod

# -----------------------------------------------------------------------------
# Feature / batch resolution
# -----------------------------------------------------------------------------
def resolve_feature_sets(df: pd.DataFrame, feature_dict_path: Path) -> Dict[str, List[str]]:
    if feature_dict_path.exists():
        fdict = load_json(feature_dict_path)
    else:
        fdict = {}
    scanner_covariates = [c for c in fdict.get("scanner_covariate_columns", []) if c in df.columns]
    all_numeric = [c for c in fdict.get("numeric_feature_columns", []) if c in df.columns]
    rad_only = [c for c in fdict.get("numeric_feature_columns_no_scanner_covariates", []) if c in df.columns]
    if not all_numeric:
        exclude = {
            "patient_base", "split", "stratum_key", "ER", "PR", "HER2",
            "ER_bin", "PR_bin", "HER2_bin", "target_er", "target_pr", "target_her2",
        }
        all_numeric = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if not rad_only:
        rad_only = [c for c in all_numeric if c not in set(scanner_covariates)]
    scanner_only = [c for c in scanner_covariates if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    # Prefer the three explicit blocks, as step 05 does. Dropping only the eight columns named
    # "scanner" leaves field strength in the analysis under four other names, plus pixel
    # spacing and slice geometry, so the 737-column "radiomics_only" set is not a radiomic arm
    # and Table S2's "(642)" label did not describe it.
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
            "_scanner_covariates": list(acq_block),
        }
    return {
        "radiomics_plus_scanner": list(all_numeric),
        "radiomics_only": list(rad_only),
        "scanner_only": list(scanner_only),
        "_radiomics_only_cols": list(rad_only),
        "_scanner_covariates": list(scanner_covariates),
    }


def get_batch_labels(df: pd.DataFrame) -> Optional[np.ndarray]:
    for col in BATCH_COLUMN_CANDIDATES:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.notna().mean() > 0.5 and vals.nunique(dropna=True) >= 2:
                return vals.fillna(-1).astype(int).values
        else:
            vals = df[col].astype(str).replace({"nan": np.nan, "None": np.nan})
            if vals.notna().mean() > 0.5 and vals.nunique(dropna=True) >= 2:
                codes = pd.Categorical(vals.fillna("missing")).codes
                return codes.astype(int)
    return None


# -----------------------------------------------------------------------------
# EB-style ComBat harmonizer
# -----------------------------------------------------------------------------
class EBCombatHarmonizer:
    """
    Dependency-light empirical-shrinkage ComBat-style harmonizer.

    The model estimates batch-specific mean and variance and shrinks them toward
    the global distribution. This is not a full neuroCombat implementation but
    behaves like an EB-style sensitivity analysis, and critically it supports
    train-only fit with separate transform for valid/test.
    """

    def __init__(self, prior_strength: float = 8.0, min_batch_size: int = 3, eps: float = 1e-8):
        self.prior_strength = float(prior_strength)
        self.min_batch_size = int(min_batch_size)
        self.eps = float(eps)
        self._fitted = False

    def fit(self, X: np.ndarray, batch: np.ndarray) -> "EBCombatHarmonizer":
        X = np.asarray(X, dtype=float)
        batch = np.asarray(batch).ravel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            gm = np.nanmean(X, axis=0)
            gs = np.nanstd(X, axis=0)
        self.grand_mean_ = np.where(np.isfinite(gm), gm, 0.0)
        self.grand_std_ = np.where(np.isfinite(gs) & (gs > self.eps), gs, 1.0)
        self.batch_stats_: Dict[Any, Dict[str, np.ndarray]] = {}

        for b in np.unique(batch):
            m = batch == b
            n_b = int(m.sum())
            if n_b < self.min_batch_size:
                self.batch_stats_[b] = {
                    "mean": self.grand_mean_.copy(),
                    "std": self.grand_std_.copy(),
                    "n": n_b,
                    "shrinkage_weight": 0.0,
                }
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                bm = np.nanmean(X[m], axis=0)
                bs = np.nanstd(X[m], axis=0)
            bm = np.where(np.isfinite(bm), bm, self.grand_mean_)
            bs = np.where(np.isfinite(bs) & (bs > self.eps), bs, self.grand_std_)
            w = n_b / (n_b + self.prior_strength)
            eb_mean = w * bm + (1.0 - w) * self.grand_mean_
            # Shrink log-variance for positivity and numerical stability.
            eb_log_std = w * np.log(bs + self.eps) + (1.0 - w) * np.log(self.grand_std_ + self.eps)
            eb_std = np.exp(eb_log_std)
            eb_std = np.where(np.isfinite(eb_std) & (eb_std > self.eps), eb_std, self.grand_std_)
            self.batch_stats_[b] = {
                "mean": eb_mean,
                "std": eb_std,
                "n": n_b,
                "shrinkage_weight": float(w),
            }
        self._fitted = True
        return self

    def transform(self, X: np.ndarray, batch: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit before transform.")
        X = np.asarray(X, dtype=float).copy()
        batch = np.asarray(batch).ravel()
        out = X.copy()
        for b in np.unique(batch):
            m = batch == b
            stats = self.batch_stats_.get(b, {"mean": self.grand_mean_, "std": self.grand_std_})
            out[m] = (out[m] - stats["mean"]) / stats["std"] * self.grand_std_ + self.grand_mean_
        return out


# -----------------------------------------------------------------------------
# Harmonization wrappers
# -----------------------------------------------------------------------------
def apply_harmonization(
    X_train: pd.DataFrame,
    X_other: pd.DataFrame,
    batch_train: np.ndarray,
    batch_other: np.ndarray,
    harm_cols: List[str],
    mode: str,
    step05_mod: Any,
    cfg: Step12Config,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if mode == "none" or not harm_cols or len(np.unique(batch_train)) < 2:
        return X_train.copy(), X_other.copy()
    cols = [c for c in harm_cols if c in X_train.columns and c in X_other.columns]
    if not cols:
        return X_train.copy(), X_other.copy()
    # Build float harmonization blocks without assigning floats into existing
    # integer columns. This prevents pandas FutureWarning and future errors.
    Xtr = X_train.copy()
    Xot = X_other.copy()
    Xtr_harm_block = Xtr.loc[:, cols].apply(pd.to_numeric, errors="coerce").astype(np.float64)
    Xot_harm_block = Xot.loc[:, cols].apply(pd.to_numeric, errors="coerce").astype(np.float64)
    Xtr = pd.concat([Xtr.drop(columns=cols), Xtr_harm_block], axis=1).reindex(columns=X_train.columns)
    Xot = pd.concat([Xot.drop(columns=cols), Xot_harm_block], axis=1).reindex(columns=X_other.columns)

    try:
        if mode == "simple_combat":
            harmonizer = step05_mod.SimpleCombatHarmonizer()
        elif mode == "eb_combat":
            harmonizer = EBCombatHarmonizer(prior_strength=cfg.eb_prior_strength)
        else:
            raise ValueError(f"Unknown harmonization mode: {mode}")
        arr_tr = Xtr.loc[:, cols].to_numpy(dtype=np.float64, copy=True)
        arr_ot = Xot.loc[:, cols].to_numpy(dtype=np.float64, copy=True)
        harmonizer.fit(arr_tr, batch_train)
        tr_h = harmonizer.transform(arr_tr, batch_train).astype(np.float64, copy=False)
        ot_h = harmonizer.transform(arr_ot, batch_other).astype(np.float64, copy=False)
        tr_h_df = pd.DataFrame(tr_h, index=Xtr.index, columns=cols, dtype=np.float64)
        ot_h_df = pd.DataFrame(ot_h, index=Xot.index, columns=cols, dtype=np.float64)
        Xtr_out = pd.concat([Xtr.drop(columns=cols), tr_h_df], axis=1).reindex(columns=X_train.columns)
        Xot_out = pd.concat([Xot.drop(columns=cols), ot_h_df], axis=1).reindex(columns=X_other.columns)
        return Xtr_out, Xot_out
    except Exception as e:
        logger.warning("%s harmonization failed (%s); using raw features.", mode, e)
        return X_train.copy(), X_other.copy()


# -----------------------------------------------------------------------------
# Metrics / bootstrap / DeLong
# -----------------------------------------------------------------------------
def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_auprc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    vals: List[float] = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(float(roc_auc_score(y[idx], p[idx])))
    if len(vals) < 50:
        return {"ci_lower": float("nan"), "ci_upper": float("nan"), "n_boot_valid": len(vals)}
    return {
        "ci_lower": float(np.percentile(vals, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(vals, 100 * (1 - alpha / 2))),
        "n_boot_valid": int(len(vals)),
    }


def paired_bootstrap_auc_difference(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    n = len(y)
    diffs: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(float(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx])))
    if len(diffs) < 50:
        return {
            "bootstrap_diff_ci95_lower": float("nan"),
            "bootstrap_diff_ci95_upper": float("nan"),
            "bootstrap_p_two_sided": float("nan"),
            "bootstrap_n_valid": len(diffs),
        }
    arr = np.asarray(diffs, dtype=float)
    n_valid = len(arr)
    # Continuity-corrected two-sided bootstrap p-value (Davison & Hinkley 1997):
    # adding 1 to numerator and denominator prevents an all-one-sided resample from
    # producing p == 0 exactly, which is statistically invalid and not publishable.
    n_le = float(np.sum(arr <= 0))
    n_ge = float(np.sum(arr >= 0))
    p_two = 2.0 * min((n_le + 1.0) / (n_valid + 1.0), (n_ge + 1.0) / (n_valid + 1.0))
    p_two = min(max(p_two, 0.0), 1.0)
    return {
        "bootstrap_diff_ci95_lower": float(np.percentile(arr, 100 * alpha / 2)),
        "bootstrap_diff_ci95_upper": float(np.percentile(arr, 100 * (1 - alpha / 2))),
        "bootstrap_p_two_sided": p_two,
        "bootstrap_n_valid": int(len(arr)),
    }


# DeLong implementation adapted from the public-domain formulation by Sun & Xu.
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    midranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        mid = 0.5 * (i + j - 1) + 1
        midranks[i:j] = mid
        i = j
    out = np.empty(n, dtype=float)
    out[order] = midranks
    return out


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int) -> Tuple[np.ndarray, np.ndarray]:
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n
    return aucs, np.atleast_2d(delong_cov)


def _calc_pvalue(aucs: np.ndarray, sigma: np.ndarray) -> float:
    diff = np.array([[1, -1]], dtype=float)
    z_num = abs(np.diff(np.asarray(aucs, dtype=float)).item())
    z_den = math.sqrt((diff @ np.asarray(sigma, dtype=float) @ diff.T).item())
    if z_den <= 0 or not np.isfinite(z_den):
        return float("nan")
    z = z_num / z_den
    # two-sided normal survival without scipy
    p = math.erfc(z / math.sqrt(2.0))
    return float(p)


def delong_roc_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Dict[str, Any]:
    y = np.asarray(y_true).astype(int)
    p1 = np.asarray(pred_a, dtype=float)
    p2 = np.asarray(pred_b, dtype=float)
    if len(np.unique(y)) < 2:
        return {"delong_p": float("nan"), "delong_status": "one_class"}
    order = np.argsort(-y)
    label_1_count = int(np.sum(y == 1))
    if label_1_count <= 1 or (len(y) - label_1_count) <= 1:
        return {"delong_p": float("nan"), "delong_status": "too_few_class_members"}
    preds = np.vstack([p1, p2])[:, order]
    try:
        aucs, cov = _fast_delong(preds, label_1_count)
        p = _calc_pvalue(aucs, cov)
        return {"delong_p": p, "delong_status": "ok", "delong_auc_a": float(aucs[0]), "delong_auc_b": float(aucs[1])}
    except Exception as e:
        return {"delong_p": float("nan"), "delong_status": f"failed: {e}"}


# -----------------------------------------------------------------------------
# Model selection / prediction generation
# -----------------------------------------------------------------------------
def get_step05_selected_model_table(cfg: Step12Config) -> pd.DataFrame:
    if Path(cfg.step05_global_best_csv).exists():
        df = pd.read_csv(cfg.step05_global_best_csv)
    elif Path(cfg.step05_results_csv).exists():
        all_results = pd.read_csv(cfg.step05_results_csv)
        df = (
            all_results.sort_values(["target", "nested_cv_auroc_mean", "nested_cv_auroc_std", "AUROC"], ascending=[True, False, True, False])
            .groupby("target", as_index=False)
            .head(1)
            .reset_index(drop=True)
        )
    else:
        raise FileNotFoundError("Step05 results not found. Run Step 05 first.")
    return df


def train_predict_locked_test(
    df: pd.DataFrame,
    feature_cols: List[str],
    harm_cols: List[str],
    batch_labels: Optional[np.ndarray],
    target_col: str,
    model_name: str,
    combat_mode: str,
    step05_mod: Any,
    cfg: Step12Config,
    logger: logging.Logger,
) -> Dict[str, Any]:
    data = df[~df[target_col].isna()].copy()
    data[target_col] = data[target_col].astype(int)
    for c in feature_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data[feature_cols] = data[feature_cols].replace([np.inf, -np.inf], np.nan)

    tv_mask = data["split"].isin(["train", "valid"])
    test_mask = data["split"] == "test"
    X_tv = data.loc[tv_mask, feature_cols].copy()
    y_tv = data.loc[tv_mask, target_col].copy()
    X_test = data.loc[test_mask, feature_cols].copy()
    y_test = data.loc[test_mask, target_col].copy()

    if len(y_tv) < 10 or len(y_test) < 4 or y_tv.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(f"Insufficient data for {target_col}: trainval={len(y_tv)}, test={len(y_test)}")

    if batch_labels is not None:
        aligned = pd.Series(batch_labels, index=df.index).reindex(data.index).fillna(-1).astype(int)
        b_tv = aligned.loc[tv_mask].values
        b_test = aligned.loc[test_mask].values
    else:
        b_tv = np.zeros(len(y_tv), dtype=int)
        b_test = np.zeros(len(y_test), dtype=int)

    X_tv_h, X_test_h = apply_harmonization(
        X_tv, X_test, b_tv, b_test, harm_cols, combat_mode, step05_mod, cfg, logger
    )

    runtime_info = step05_mod.detect_runtime_environment()
    gpu = bool(runtime_info.get("gpu_available", False))
    candidates = step05_mod.build_candidate_models(step05_mod.Step05Config(), y_tv, gpu)
    if model_name not in candidates:
        # Robust fallback when LightGBM/XGB not installed.
        logger.warning("Model %s unavailable; falling back to elastic_net.", model_name)
        model_name = "elastic_net"
    model = candidates[model_name]
    model.fit(X_tv_h, y_tv)

    probs = model.predict_proba(X_test_h)[:, 1]
    oof_thr = 0.5
    try:
        oof_thr = float(step05_mod.youden_threshold(y_tv.values, model.predict_proba(X_tv_h)[:, 1]))
    except Exception:
        oof_thr = 0.5

    return {
        "target": target_col,
        "target_display": TARGET_DISPLAY.get(target_col, target_col),
        "model": model_name,
        "combat_mode": combat_mode,
        "n_trainval": int(len(y_tv)),
        "n_test": int(len(y_test)),
        "n_features": int(len(feature_cols)),
        "y_test": y_test.to_numpy(dtype=int),
        "probs": probs,
        "patient_ids": data.loc[test_mask, "patient_base"].to_numpy() if "patient_base" in data.columns else np.arange(len(y_test)),
        "threshold": oof_thr,
        "AUROC": safe_auc(y_test.to_numpy(), probs),
        "AUPRC": safe_auprc(y_test.to_numpy(), probs),
    }


def run_ebcombat_sensitivity(
    df: pd.DataFrame,
    feature_sets: Dict[str, List[str]],
    batch_labels: Optional[np.ndarray],
    step05_mod: Any,
    selected_models: pd.DataFrame,
    cfg: Step12Config,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []
    harm_cols = feature_sets.get("_radiomics_only_cols", [])

    for target_col in cfg.target_columns:
        if target_col not in df.columns:
            continue
        # Prefer Step05 global best model for the target; if absent, use elastic_net.
        sel = selected_models[selected_models["target"] == target_col]
        default_model = str(sel.iloc[0]["model"]) if not sel.empty and "model" in sel.columns else "elastic_net"

        for variant in cfg.feature_variants:
            feature_cols = feature_sets.get(variant, [])
            if not feature_cols:
                continue
            for mode in cfg.combat_modes:
                # Scanner-only or no radiomics should not be ComBat harmonized.
                if variant in ("scanner_only", "acquisition_only") and mode != "none":
                    continue
                try:
                    out = train_predict_locked_test(
                        df=df,
                        feature_cols=feature_cols,
                        harm_cols=harm_cols,
                        batch_labels=batch_labels,
                        target_col=target_col,
                        model_name=default_model,
                        combat_mode=mode,
                        step05_mod=step05_mod,
                        cfg=cfg,
                        logger=logger,
                    )
                    ci = bootstrap_auc_ci(out["y_test"], out["probs"], cfg.bootstrap_n, cfg.global_seed, cfg.bootstrap_alpha)
                    row = {
                        k: v for k, v in out.items()
                        if k not in {"y_test", "probs", "patient_ids"}
                    }
                    row.update({
                        "feature_set_variant": variant,
                        "AUROC_CI95_lower": ci["ci_lower"],
                        "AUROC_CI95_upper": ci["ci_upper"],
                        "bootstrap_n_valid": ci["n_boot_valid"],
                    })
                    rows.append(row)
                    for pid, y, p in zip(out["patient_ids"], out["y_test"], out["probs"]):
                        pred_rows.append({
                            "scope": "internal_locked_test",
                            "target": target_col,
                            "target_display": TARGET_DISPLAY.get(target_col, target_col),
                            "feature_set_variant": variant,
                            "combat_mode": mode,
                            "model": row["model"],
                            "patient_id": pid,
                            "y_true": int(y),
                            "probability": float(p),
                        })
                    logger.info(
                        "Step12 sensitivity | %s | %s | %s | model=%s | AUROC=%.3f",
                        target_col, variant, mode, row["model"], row["AUROC"],
                    )
                except Exception as e:
                    logger.warning("Sensitivity failed target=%s variant=%s mode=%s: %s", target_col, variant, mode, e)
                    rows.append({
                        "target": target_col,
                        "target_display": TARGET_DISPLAY.get(target_col, target_col),
                        "feature_set_variant": variant,
                        "combat_mode": mode,
                        "model": default_model,
                        "status": f"failed: {e}",
                    })
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


# -----------------------------------------------------------------------------
# Pairwise comparisons
# -----------------------------------------------------------------------------
def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values (no scipy dependency).

    NaN p-values are passed through unchanged and excluded from the ranking and
    from the count m. Returns an array aligned to the input order.
    """
    p = np.asarray(pvals, dtype=float)
    adj = np.full(p.shape, np.nan, dtype=float)
    finite = np.where(np.isfinite(p))[0]
    m = finite.size
    if m == 0:
        return adj
    order = finite[np.argsort(p[finite])]
    ranked = p[order]
    adj_sorted = ranked * m / np.arange(1, m + 1)
    # enforce monotonicity from the largest p-value downwards
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adj[order] = np.clip(adj_sorted, 0.0, 1.0)
    return adj


def compare_prediction_groups(pred_df: pd.DataFrame, cfg: Step12Config, logger: logging.Logger) -> pd.DataFrame:
    if pred_df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    group_cols = ["scope", "target"]
    for (scope, target), g in pred_df.groupby(group_cols, dropna=False):
        # Model identity for comparison: variant + combat + model.
        g = g.copy()
        g["comparison_id"] = (
            g.get("feature_set_variant", pd.Series([""] * len(g))).astype(str)
            + "|" + g.get("combat_mode", pd.Series([""] * len(g))).astype(str)
            + "|" + g.get("model", pd.Series([""] * len(g))).astype(str)
        )
        ids = sorted(g["comparison_id"].dropna().unique().tolist())
        if len(ids) < 2:
            continue
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                ga = g[g["comparison_id"] == a][["patient_id", "y_true", "probability"]].copy()
                gb = g[g["comparison_id"] == b][["patient_id", "y_true", "probability"]].copy()
                merged = ga.merge(gb, on=["patient_id", "y_true"], how="inner", suffixes=("_a", "_b"))
                if merged.shape[0] < 4 or merged["y_true"].nunique() < 2:
                    continue
                y = merged["y_true"].to_numpy(dtype=int)
                pa = merged["probability_a"].to_numpy(dtype=float)
                pb = merged["probability_b"].to_numpy(dtype=float)
                auc_a = safe_auc(y, pa)
                auc_b = safe_auc(y, pb)
                boot = paired_bootstrap_auc_difference(y, pa, pb, cfg.bootstrap_n, cfg.global_seed, cfg.bootstrap_alpha)
                dl = delong_roc_test(y, pa, pb)
                rows.append({
                    "scope": scope,
                    "target": target,
                    "target_display": TARGET_DISPLAY.get(target, target),
                    "model_a": a,
                    "model_b": b,
                    "n_paired": int(len(y)),
                    "n_positive": int(np.sum(y == 1)),
                    "n_negative": int(np.sum(y == 0)),
                    "AUROC_a": auc_a,
                    "AUROC_b": auc_b,
                    "AUROC_diff_a_minus_b": float(auc_a - auc_b) if np.isfinite(auc_a) and np.isfinite(auc_b) else float("nan"),
                    **boot,
                    **dl,
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        # Multiple-comparison correction: Benjamini-Hochberg FDR is applied within
        # each (scope, target) family — the set of pairwise comparisons that share a
        # hypothesis. Raw p-values are retained alongside the adjusted columns.
        adj_pairs = [
            ("bootstrap_p_two_sided", "bootstrap_p_two_sided_fdr_bh"),
            ("delong_p", "delong_p_fdr_bh"),
        ]
        for _, adjcol in adj_pairs:
            out[adjcol] = np.nan
        for _, sub_idx in out.groupby(["scope", "target"], dropna=False).groups.items():
            for pcol, adjcol in adj_pairs:
                if pcol in out.columns:
                    out.loc[sub_idx, adjcol] = benjamini_hochberg(
                        out.loc[sub_idx, pcol].to_numpy(dtype=float)
                    )
        n_families = out.groupby(["scope", "target"], dropna=False).ngroups
        logger.info(
            "Pairwise comparisons: %d rows across %d (scope,target) families; "
            "applied Benjamini-Hochberg FDR within each family.",
            len(out), n_families,
        )
        out = out.sort_values(["scope", "target", "AUROC_diff_a_minus_b"], ascending=[True, True, False]).reset_index(drop=True)
    return out


def load_external_predictions_for_comparison(cfg: Step12Config, logger: logging.Logger) -> pd.DataFrame:
    """Load Step 11 external predictions and normalize to Step 12 schema.

    Pairwise comparison requires at least two prediction sources per target. When
    Step 11 provides one source only, main() writes a clear placeholder table.
    """
    p = Path(cfg.step11_predictions_csv)
    if not p.exists():
        logger.info("Step11 prediction file not found; skipping external pairwise comparison: %s", p)
        return pd.DataFrame()

    df = pd.read_csv(p)
    if df.empty:
        logger.info("Step11 prediction file is empty: %s", p)
        return pd.DataFrame()

    lower_to_original = {c.lower(): c for c in df.columns}

    def pick(candidates: Sequence[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
            if c.lower() in lower_to_original:
                return lower_to_original[c.lower()]
        return None

    target_col = pick(["target", "target_col", "target_name"])
    patient_col = pick(["patient_id", "patient", "openbtai_patient_id", "patient_base"])
    y_col = pick(["y_true", "y_true_proxy", "label", "external_label", "proxy_label", "true_label"])
    p_col = pick(["probability", "prob", "pred_prob", "prediction_probability", "y_score"])
    display_col = pick(["target_display", "target_label", "target_name"])
    variant_col = pick(["feature_set_variant", "feature_variant", "variant"])
    combat_col = pick(["combat_mode", "harmonization", "combat"])
    model_col = pick(["model", "model_name", "classifier"])

    required_missing = [name for name, col in {
        "target": target_col,
        "patient_id": patient_col,
        "y_true": y_col,
        "probability": p_col,
    }.items() if col is None]
    if required_missing:
        logger.warning("Step11 predictions missing required columns %s; found %s", required_missing, list(df.columns))
        return pd.DataFrame()

    out = pd.DataFrame({
        "scope": "external_openbtai_proxy",
        "target": df[target_col].astype(str),
        "target_display": df[display_col].astype(str) if display_col else df[target_col].astype(str),
        "feature_set_variant": df[variant_col].astype(str) if variant_col else "external_proxy",
        "combat_mode": df[combat_col].astype(str) if combat_col else "step11",
        "model": df[model_col].astype(str) if model_col else "step11_model",
        "patient_id": df[patient_col].astype(str),
        "y_true": pd.to_numeric(df[y_col], errors="coerce"),
        "probability": pd.to_numeric(df[p_col], errors="coerce"),
    })
    out = out.dropna(subset=["y_true", "probability"]).copy()
    out["y_true"] = out["y_true"].astype(int)
    out["probability"] = out["probability"].astype(float)
    return out


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def save_sensitivity_barplot(results_df: pd.DataFrame, path: Path) -> None:
    if results_df.empty or "AUROC" not in results_df.columns:
        return
    ok = results_df[pd.to_numeric(results_df["AUROC"], errors="coerce").notna()].copy()
    if ok.empty:
        return
    ok["label"] = ok["target_display"].astype(str) + "\n" + ok["feature_set_variant"].astype(str) + "\n" + ok["combat_mode"].astype(str)
    plt.figure(figsize=(max(12, len(ok) * 0.6), 6))
    x = np.arange(len(ok))
    vals = ok["AUROC"].astype(float).to_numpy()
    plt.bar(x, vals)
    if {"AUROC_CI95_lower", "AUROC_CI95_upper"}.issubset(ok.columns):
        lo = ok["AUROC_CI95_lower"].astype(float).to_numpy()
        hi = ok["AUROC_CI95_upper"].astype(float).to_numpy()
        yerr = np.vstack([vals - lo, hi - vals])
        yerr = np.where(np.isfinite(yerr), np.maximum(yerr, 0), 0)
        plt.errorbar(x, vals, yerr=yerr, fmt="none", capsize=3)
    plt.axhline(0.5, linestyle="--", linewidth=1)
    plt.ylim(0, 1.05)
    plt.xticks(x, ok["label"].tolist(), rotation=70, ha="right")
    plt.ylabel("Locked-test AUROC")
    plt.title("Step 12 EB-ComBat sensitivity")
    plt.tight_layout()
    ensure_dir(path.parent)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_pairwise_heatmap(comp_df: pd.DataFrame, path: Path, title: str) -> None:
    if comp_df.empty:
        return
    ok = comp_df[pd.to_numeric(comp_df["AUROC_diff_a_minus_b"], errors="coerce").notna()].copy()
    if ok.empty:
        return
    labels = (ok["target_display"].astype(str) + ": " + ok["model_a"].astype(str) + " vs " + ok["model_b"].astype(str)).tolist()
    vals = ok["AUROC_diff_a_minus_b"].astype(float).to_numpy()
    plt.figure(figsize=(max(10, len(vals) * 0.6), 5))
    x = np.arange(len(vals))
    plt.bar(x, vals)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(x, labels, rotation=70, ha="right")
    plt.ylabel("AUROC difference")
    plt.title(title)
    plt.tight_layout()
    ensure_dir(path.parent)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    _ap = argparse.ArgumentParser(add_help=False)
    _ap.add_argument("--force",    action="store_true")
    _ap.add_argument("--no-cache", action="store_true")
    _flags, _ = _ap.parse_known_args()

    cfg = Step12Config()
    set_global_seed(cfg.global_seed)
    for p in [REPORTS_DIR, TABLES_DIR, FIGURES_DIR, METADATA_DIR, LOGS_DIR, CHECKPOINTS_DIR]:
        ensure_dir(p)
    logger = setup_logger(LOGS_DIR / f"{STEP_NAME}.log")
    logger.info("Starting %s", STEP_NAME)

    # ── Cache check ───────────────────────────────────────────────────────────
    _cfg_hash = hashlib.md5(json.dumps({
        "seed": cfg.global_seed, "n_bootstrap": cfg.bootstrap_n,
        "combat_modes": sorted(cfg.combat_modes), "targets": sorted(cfg.target_columns),
    }, sort_keys=True).encode()).hexdigest()
    _manifest = PROJECT_ROOT / "cache" / "step12_cache_manifest.json"
    _inputs   = [INPUT_PATIENT_LEVEL, STEP05_RESULTS]
    _outputs  = [TABLES_DIR / "step12_ebcombat_sensitivity_results.csv", METADATA_DIR / "step12_summary.json"]
    if not _flags.force and not _flags.no_cache and _is_step_cached(_manifest, _inputs, _cfg_hash, _outputs):
        logger.info("Cache valid — Step 12 outputs unchanged. Skipping (use --force to re-run).")
        return

    if not INPUT_PATIENT_LEVEL.exists():
        raise FileNotFoundError(f"Missing Step04 patient-level file: {INPUT_PATIENT_LEVEL}")
    df = pd.read_csv(INPUT_PATIENT_LEVEL)
    logger.info("Loaded BCBM patient-level: %s", df.shape)
    if "split" not in df.columns:
        raise ValueError("Missing 'split' column in Step04 patient-level file.")

    step05 = import_step05_module()
    feature_sets = resolve_feature_sets(df, INPUT_FEATURE_DICT)
    batch_labels = get_batch_labels(df)
    selected_models = get_step05_selected_model_table(cfg)
    logger.info("Loaded Step05 selected models: %s", selected_models.shape)
    logger.info("Feature set sizes: %s", {k: len(v) for k, v in feature_sets.items() if not k.startswith('_')})
    logger.info("Batch labels available: %s", batch_labels is not None)

    summary: Dict[str, Any] = {
        "config": asdict(cfg),
        "input_shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "feature_set_sizes": {k: len(v) for k, v in feature_sets.items() if not k.startswith("_")},
        "batch_labels_available": bool(batch_labels is not None),
        "outputs": {},
        "method_notes": {
            "eb_combat": "Empirical-shrinkage ComBat-style sensitivity; fit on train/trainval only.",
            "statistical_comparison": "DeLong test plus paired bootstrap AUROC difference; paired bootstrap preferred for small cohorts.",
        },
    }

    sensitivity_df = pd.DataFrame()
    internal_pred_df = pd.DataFrame()
    if cfg.run_ebcombat_sensitivity:
        sensitivity_df, internal_pred_df = run_ebcombat_sensitivity(
            df=df,
            feature_sets=feature_sets,
            batch_labels=batch_labels,
            step05_mod=step05,
            selected_models=selected_models,
            cfg=cfg,
            logger=logger,
        )
        sensitivity_csv = TABLES_DIR / "step12_ebcombat_sensitivity_results.csv"
        predictions_csv = TABLES_DIR / "step12_locked_test_predictions.csv"
        sensitivity_df.to_csv(sensitivity_csv, index=False, encoding="utf-8-sig")
        internal_pred_df.to_csv(predictions_csv, index=False, encoding="utf-8-sig")
        summary["outputs"]["ebcombat_sensitivity_results_csv"] = str(sensitivity_csv)
        summary["outputs"]["locked_test_predictions_csv"] = str(predictions_csv)
        if cfg.save_plots:
            fig_path = FIGURES_DIR / "step12_ebcombat_sensitivity_auroc.png"
            save_sensitivity_barplot(sensitivity_df, fig_path)
            summary["outputs"]["ebcombat_sensitivity_figure"] = str(fig_path)

    internal_comp_df = pd.DataFrame()
    if cfg.run_internal_pairwise_comparison and not internal_pred_df.empty:
        internal_comp_df = compare_prediction_groups(internal_pred_df, cfg, logger)
        comp_csv = TABLES_DIR / "step12_internal_pairwise_model_comparison.csv"
        internal_comp_df.to_csv(comp_csv, index=False, encoding="utf-8-sig")
        summary["outputs"]["internal_pairwise_comparison_csv"] = str(comp_csv)
        if cfg.save_plots:
            fig_path = FIGURES_DIR / "step12_internal_pairwise_auc_differences.png"
            save_pairwise_heatmap(internal_comp_df, fig_path, "Internal locked-test paired AUROC differences")
            summary["outputs"]["internal_pairwise_figure"] = str(fig_path)

    external_comp_df = pd.DataFrame()
    if cfg.run_external_pairwise_comparison:
        ext_pred = load_external_predictions_for_comparison(cfg, logger)
        # Pairwise external comparison is only possible if Step11 contains more than one
        # prediction source per target. If not, save a clear placeholder table.
        if ext_pred.empty:
            external_comp_df = pd.DataFrame([{"status": "skipped_no_step11_predictions_or_single_model"}])
        else:
            external_comp_df = compare_prediction_groups(ext_pred, cfg, logger)
            if external_comp_df.empty:
                external_comp_df = pd.DataFrame([{
                    "status": "skipped_single_external_prediction_source_per_target",
                    "note": "Step11 currently provides one external prediction source per target, so paired model comparison is not applicable.",
                }])
        ext_csv = TABLES_DIR / "step12_external_pairwise_model_comparison.csv"
        external_comp_df.to_csv(ext_csv, index=False, encoding="utf-8-sig")
        summary["outputs"]["external_pairwise_comparison_csv"] = str(ext_csv)

    summary["n_sensitivity_rows"] = int(sensitivity_df.shape[0])
    summary["n_internal_prediction_rows"] = int(internal_pred_df.shape[0])
    summary["n_internal_pairwise_rows"] = int(internal_comp_df.shape[0])
    summary["n_external_pairwise_rows"] = int(external_comp_df.shape[0])

    summary_json = METADATA_DIR / "step12_summary.json"
    summary["outputs"]["summary_json"] = str(summary_json)
    save_json(summary, summary_json)

    if cfg.save_zip:
        zip_path = REPORTS_DIR / "step12_combat_eb_statistical_comparison_package.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(TABLES_DIR.glob("step12_*")):
                zf.write(p, arcname=f"tables/{p.name}")
            for p in sorted(FIGURES_DIR.glob("step12_*")):
                zf.write(p, arcname=f"figures/{p.name}")
            zf.write(summary_json, arcname=f"metadata/{summary_json.name}")
        summary["outputs"]["publication_zip"] = str(zip_path)
        save_json(summary, summary_json)

    # Update pipeline state without requiring the existing StateManager.
    state_path = CHECKPOINTS_DIR / "pipeline_state.json"
    if state_path.exists():
        try:
            state = load_json(state_path)
        except Exception:
            state = {"steps_completed": [], "artifacts": {}, "notes": []}
    else:
        state = {"steps_completed": [], "artifacts": {}, "notes": []}
    state.setdefault("steps_completed", [])
    if STEP_NAME not in state["steps_completed"]:
        state["steps_completed"].append(STEP_NAME)
    state.setdefault("artifacts", {}).update(summary.get("outputs", {}))
    state.setdefault("notes", []).append({"note": "Step 12 EB-ComBat sensitivity and statistical comparison completed."})
    save_json(state, state_path)

    _save_step_manifest(_manifest, _inputs, _cfg_hash)
    logger.info("Completed %s successfully.", STEP_NAME)
    logger.info("Summary: %s", summary_json)
    logger.info("=" * 80)

    if not sensitivity_df.empty:
        display_cols = [c for c in ["target_display", "feature_set_variant", "combat_mode", "model", "n_test", "AUROC", "AUROC_CI95_lower", "AUROC_CI95_upper", "AUPRC"] if c in sensitivity_df.columns]
        print("\nSTEP 12 EB-ComBat sensitivity")
        print(sensitivity_df[display_cols].to_string(index=False))
    if not internal_comp_df.empty:
        display_cols = [c for c in ["target_display", "model_a", "model_b", "n_paired", "AUROC_a", "AUROC_b", "AUROC_diff_a_minus_b", "bootstrap_diff_ci95_lower", "bootstrap_diff_ci95_upper", "bootstrap_p_two_sided", "delong_p", "delong_status"] if c in internal_comp_df.columns]
        print("\nSTEP 12 Internal pairwise statistical comparison")
        print(internal_comp_df[display_cols].head(40).to_string(index=False))


if __name__ == "__main__":
    main()
