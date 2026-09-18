from __future__ import annotations

"""
Step 13 — Integrated Publication Figures and Reports
=====================================================

This is the final reporting step. It aggregates results from all prior steps
(05–12) and produces the publication-grade figure and table package.

File location:
    src/reporting/step_13_figures_and_reports.py

It is designed to work with the latest methodology updates:

- Step 05: nested CV / ComBat-safe / stability-selected tabular modeling
- Step 07: MIL radiomics with validation-based model selection
- Step 08: strong 2.5D CNN image modeling
- Step 09: strong hybrid fusion

Outputs:
    reports/figures/step13_publication_package/*.png
    reports/figures/step13_publication_package/*.pdf
    reports/figures/step13_publication_package/*.svg for graphical abstract
    reports/figures/step13_publication_package/figure_manifest.csv
    reports/step13_publication_figures.zip
    metadata/step13_publication_summary.json

The script is intentionally fault-tolerant: one failed figure does not stop the
remaining figures.
"""

import argparse
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import textwrap
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except Exception:
    sns = None
    SEABORN_AVAILABLE = False

try:
    from scipy.stats import gaussian_kde, mannwhitneyu
    from scipy.special import betainc
    SCIPY_AVAILABLE = True
except Exception:
    gaussian_kde = None
    mannwhitneyu = None
    betainc = None
    SCIPY_AVAILABLE = False

# ── Project paths ────────────────────────────────────────────────────────────
# Anchor to the repo root (two levels above src/reporting/) instead of the current
# working directory, so the step resolves correct input/output paths even when run
# standalone. Launched via run_pipeline.py (cwd = repo root) this is identical; the
# previous Path("./bcbm_project") silently loaded every input as empty when run from
# any other directory, degrading all figures to NaN/placeholders without error.
_REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = _REPO_ROOT / "bcbm_project"
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = Path("./bcbm_project").resolve()  # legacy cwd-relative fallback
STEP_NAME = "step_13_figures_and_reports"

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
METADATA_DIR = PROJECT_ROOT / "metadata"
LOGS_DIR = PROJECT_ROOT / "logs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = FIGURES_DIR / "step13_publication_package"

INPUT_STEP03 = PROCESSED_DIR / "analysis_ready_step03_lesion_only.csv"
INPUT_STEP04 = PROCESSED_DIR / "analysis_ready_step04_patient_level.csv"
INPUT_FEATURE_DICT = METADATA_DIR / "step04_feature_dictionary.json"

# Main artifacts from previous steps. Multiple candidates preserve backward compatibility.
STEP04_SUMMARY_CANDIDATES = [METADATA_DIR / "step04_summary.json", METADATA_DIR / "step04_feature_engineering_summary.json"]
STEP05_SUMMARY_CANDIDATES = [METADATA_DIR / "step05master_summary.json", METADATA_DIR / "step05_summary.json"]
STEP05_RESULTS_CANDIDATES = [
    TABLES_DIR / "step05master_model_results.csv",
    TABLES_DIR / "step05_advanced_v3_model_results.csv",
    TABLES_DIR / "step05_model_results.csv",
]
STEP05_BEST_CANDIDATES = [
    TABLES_DIR / "step05master_global_best_models_by_target.csv",
    TABLES_DIR / "step05master_best_models_by_target_and_scenario.csv",
    TABLES_DIR / "step05_best_models_by_target.csv",
]
STEP05_VARIANT_CANDIDATES = [TABLES_DIR / "step05master_feature_set_variant_summary.csv", TABLES_DIR / "step05_feature_set_variant_summary.csv"]
STEP05_VALIDATION_CANDIDATES = [
    TABLES_DIR / "step05master_validation_model_comparison_master_table.csv",
    TABLES_DIR / "step05master_inner_cv_comparison_master.csv",
    TABLES_DIR / "step05_validation_model_comparison_master_table.csv",
]
STEP07_SUMMARY_CANDIDATES = [METADATA_DIR / "step07_mil_summary.json"]
STEP07_RESULTS_CANDIDATES = [TABLES_DIR / "step07_mil_model_results.csv", TABLES_DIR / "step07_mil_best_by_target.csv"]
STEP08_RESULTS_CANDIDATES = [TABLES_DIR / "step08_cnn_model_results.csv"]
STEP09_RESULTS_CANDIDATES = [TABLES_DIR / "step09_hybrid_model_results.csv"]
STEP09_COMPARISON_CANDIDATES = [TABLES_DIR / "step09_final_comparison_all_stages.csv"]
# Cross-fitted out-of-fold scores from the repeated nested CV: one averaged probability per
# patient. Figures S4/S5 need these; without them a real ROC curve cannot be drawn at all.
STEP05B_SCORE_CANDIDATES = [
    TABLES_DIR / "step05b_crossfitted_scores_rocscores.csv",
    TABLES_DIR / "step05b_crossfitted_scores_primary.csv",
]

# ── Constants ────────────────────────────────────────────────────────────────
GLOBAL_SEED = 42
TARGET_COLUMNS = ["target_er", "target_pr", "target_her2"]
TARGET_DISPLAY = {"target_er": "ER", "target_pr": "PR", "target_her2": "HER2"}
RAW_LABEL_MAP = {"target_er": "ER", "target_pr": "PR", "target_her2": "HER2"}
TARGET_COLORS = {"target_er": "#E07A5F", "target_pr": "#048A81", "target_her2": "#2E4057"}
PALETTE = ["#2E4057", "#048A81", "#E07A5F", "#F2CC8F", "#54C6EB"]
GRAY = "#7A7A7A"
LIGHT_GRAY = "#E8E8E8"

FEATURE_FAMILY_COLORS = {
    "FirstOrder": "#E07A5F",
    "Shape": "#9467BD",
    "GLCM": "#2E4057",
    "GLRLM": "#048A81",
    "GLSZM": "#4CAF50",
    "GLDM": "#F2CC8F",
    "NGTDM": "#54C6EB",
    "Wavelet": "#8D6E63",
    "Other": "#9E9E9E",
}

VARIANT_DISPLAY = {
    "all_features": "All features",
    "radiomics_pure": "Radiomics only",
    "radiomics_plus_burden": "+ burden/study",
    "acquisition_only": "Acquisition only",
    "legacy_single_variant": "Legacy",
}

# ── Global style ─────────────────────────────────────────────────────────────
def setup_global_style() -> None:
    if SEABORN_AVAILABLE:
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except Exception:
            try:
                plt.style.use("seaborn-whitegrid")
            except Exception:
                pass
    plt.rcParams.update({
        "font.family": ["DejaVu Sans", "Arial", "sans-serif"],
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


@dataclass
class Step10Config:
    project_root: str = str(PROJECT_ROOT)
    step_name: str = STEP_NAME
    output_dir: str = str(OUTPUT_DIR)
    global_seed: int = GLOBAL_SEED
    save_zip: bool = True
    dpi: int = 300
    max_heatmap_features: int = 20
    max_violin_features: int = 6


class StateManager:
    def __init__(self, state_path: Path):
        self.state_path = state_path
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"steps_completed": [], "artifacts": {}, "notes": []}

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2, ensure_ascii=False), encoding="utf-8")

    def mark_step_done(self, step_name: str, artifacts: Optional[Dict[str, str]] = None) -> None:
        if step_name not in self.state.setdefault("steps_completed", []):
            self.state["steps_completed"].append(step_name)
        if artifacts:
            self.state.setdefault("artifacts", {}).update(artifacts)
        self.save()

    def add_note(self, note: str) -> None:
        self.state.setdefault("notes", []).append({"note": note})
        self.save()


# ── Helpers ─────────────────────────────────────────────────────────────────
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger(STEP_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    sh = logging.StreamHandler()
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    from datetime import datetime
    logger.info("=" * 80)
    logger.info("NEW SESSION STARTED at %s", datetime.utcnow().isoformat())
    logger.info("=" * 80)
    return logger


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


def load_json_safe(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logging.getLogger(STEP_NAME).warning("Failed to parse JSON %s: %s", path, e)
        return {}


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def first_existing(candidates: Sequence[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def read_csv_first(candidates: Sequence[Path]) -> pd.DataFrame:
    p = first_existing(candidates)
    if p is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception as e:
        logging.getLogger(STEP_NAME).warning("Failed to read CSV %s: %s", p, e)
        return pd.DataFrame()


def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(round(float(x)))
    except Exception:
        return default


_IBSI_CANONICAL: Optional[Dict[str, str]] = None


def _ibsi_canonical() -> Dict[str, str]:
    """lowercased IBSI feature name -> the spelling PyRadiomics actually uses.

    Step 04 lowercases every column name, so a patient-level column reads
    `patient_original_glszm_sizezonenonuniformitynormalized_mean`. Printed as-is it wraps
    mid-word ("Sizezonenonuni / formitynormalized"), which is not a feature name a reader can
    recognise. The camel case still exists in the step-03 lesion table, so it is restored from
    there; if that table is unavailable the name is printed as stored.
    """
    global _IBSI_CANONICAL
    if _IBSI_CANONICAL is None:
        _IBSI_CANONICAL = {}
        try:
            cols = pd.read_csv(INPUT_STEP03, nrows=0).columns
            for c in cols:
                if c.startswith("original_"):
                    _IBSI_CANONICAL[c.lower()] = c
        except Exception:
            pass
    return _IBSI_CANONICAL


def clean_feature_name(name: str) -> str:
    s = str(name)
    if s.startswith("patient_original_"):
        # patient_original_<feature>_<summary> -> restore the canonical <feature>
        stem, _, summary = s[len("patient_"):].rpartition("_")
        s = "%s_%s" % (_ibsi_canonical().get(stem, stem), summary)
    for prefix in ["patient_original_", "patient_", "original_", "wavelet_", "log_sigma_"]:
        if s.startswith(prefix):
            s = s[len(prefix):]
    # Split the IBSI camel case before lowering it, so the words survive .title().
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)
    s = s.replace("_", " ").replace("-", " ")
    # Wide enough that the patient-level summary (Mean, Min, Max, ...) is not truncated away.
    return textwrap.shorten(s.title(), width=56, placeholder="…")


def get_feature_family(name: str) -> str:
    s = str(name).lower()
    if "firstorder" in s or "first order" in s:
        return "FirstOrder"
    if "shape" in s:
        return "Shape"
    if "glcm" in s:
        return "GLCM"
    if "glrlm" in s:
        return "GLRLM"
    if "glszm" in s:
        return "GLSZM"
    if "gldm" in s:
        return "GLDM"
    if "ngtdm" in s:
        return "NGTDM"
    if "wavelet" in s:
        return "Wavelet"
    return "Other"


def variant_name(x: Any) -> str:
    return VARIANT_DISPLAY.get(str(x), str(x))


def save_figure(fig: plt.Figure, output_dir: Path, filename: str, dpi: int = 300, svg: bool = False) -> str:
    ensure_dir(output_dir)
    png_path = output_dir / f"{filename}.png"
    pdf_path = output_dir / f"{filename}.pdf"
    # Always close the figure, even if savefig raises (disk full, bad path), so we
    # do not leak figures across the ~20-figure generation loop.
    try:
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
        if svg:
            fig.savefig(output_dir / f"{filename}.svg", bbox_inches="tight")
    finally:
        plt.close(fig)
    return str(png_path)


def standard_result(success: bool, output_path: Optional[str] = None, reason: Optional[str] = None, section: str = "") -> Dict[str, Any]:
    d = {"success": bool(success), "section": section}
    if output_path:
        d["output_path"] = output_path
    if reason:
        d["reason"] = reason
    return d


def run_figure(fn: Callable[[Dict[str, Any], Path, logging.Logger], Dict[str, Any]], data: Dict[str, Any], output_dir: Path, logger: logging.Logger, name: str) -> Dict[str, Any]:
    try:
        logger.info("Generating %s", name)
        return fn(data, output_dir, logger)
    except Exception as e:
        logger.warning("%s failed: %s", name, e)
        return standard_result(False, reason=str(e))


def extract_metric(row: pd.Series, names: Sequence[str], default: float = np.nan) -> float:
    for n in names:
        if n in row.index:
            v = safe_float(row.get(n), np.nan)
            if not np.isnan(v):
                return v
    return default


def ensure_result_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    if "feature_set_variant" not in d.columns:
        d["feature_set_variant"] = "legacy_single_variant"
    if "scenario" not in d.columns:
        d["scenario"] = "current"
    if "model_display" not in d.columns:
        d["model_display"] = d.get("model", pd.Series(["model"] * len(d))).astype(str)
    # Normalize common Step 05/07/08/09 metric columns to base names where possible.
    metric_map = {
        "AUROC": ["AUROC", "test_AUROC", "Test_AUROC", "nested_cv_auroc_mean"],
        "AUPRC": ["AUPRC", "test_AUPRC", "Test_AUPRC"],
        "Balanced_Acc": ["Balanced_Acc", "test_Balanced_Acc", "Test_BA"],
        "F1": ["F1", "test_F1", "Test_F1"],
        "Sensitivity": ["Sensitivity", "Recall_Sensitivity", "test_Sensitivity"],
        "Specificity": ["Specificity", "test_Specificity"],
        "PPV_Precision": ["PPV_Precision", "Precision", "test_PPV_Precision", "test_Precision"],
        "NPV": ["NPV", "test_NPV"],
        "TP": ["TP", "test_TP"],
        "TN": ["TN", "test_TN"],
        "FP": ["FP", "test_FP"],
        "FN": ["FN", "test_FN"],
        "validation_AUROC_at_selection": ["validation_AUROC_at_selection", "valid_AUROC", "nested_cv_auroc_mean", "mean_inner_cv_auroc"],
        "generalization_gap_AUROC": ["generalization_gap_AUROC", "test_generalization_gap", "generalization_gap"],
        "n_selected_features": ["n_selected_features", "features", "n_features"],
    }
    for out, candidates in metric_map.items():
        if out not in d.columns:
            d[out] = [extract_metric(row, candidates) for _, row in d.iterrows()]
    return d


def summary_to_results_df(summary_json: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    targets = summary_json.get("targets", {})
    if not isinstance(targets, dict):
        return pd.DataFrame()
    for target, payload in targets.items():
        if not isinstance(payload, dict):
            continue
        # New Step 05 structure: targets[target][variant] or old: targets[target][scenario][variant]
        for key1, val1 in payload.items():
            if not isinstance(val1, dict):
                continue
            if "best_model_test_metrics" in val1:
                row = dict(val1.get("best_model_test_metrics", {}))
                row.setdefault("target", target)
                row.setdefault("feature_set_variant", key1)
                row.setdefault("scenario", "current")
                rows.append(row)
            else:
                scenario = key1
                for variant, val2 in val1.items():
                    if isinstance(val2, dict) and "best_model_test_metrics" in val2:
                        row = dict(val2.get("best_model_test_metrics", {}))
                        row.setdefault("target", target)
                        row.setdefault("feature_set_variant", variant)
                        row.setdefault("scenario", scenario)
                        rows.append(row)
    return ensure_result_columns(pd.DataFrame(rows))


IMPORTANCE_VARIANT = "radiomics_pure"   # the arm Figures S7, S8 and S12 describe


def _restrict_to_variant(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Keep one feature-set variant so importances within a column share a model and a scale."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    if "feature_set_variant" not in df.columns:
        return df
    sub = df[df["feature_set_variant"].astype(str) == variant]
    return sub.reset_index(drop=True) if not sub.empty else df


def summary_to_importance_df(summary_json: Dict[str, Any], kind: str = "top_native_features") -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    targets = summary_json.get("targets", {})
    if not isinstance(targets, dict):
        return pd.DataFrame()
    for target, payload in targets.items():
        if not isinstance(payload, dict):
            continue
        def add_items(items: Any, scenario: str, variant: str):
            if not isinstance(items, list):
                return
            for it in items:
                if isinstance(it, dict):
                    rows.append({
                        "target": target,
                        "scenario": scenario,
                        "feature_set_variant": variant,
                        "feature": it.get("feature"),
                        "importance": safe_float(it.get("importance", 0.0), 0.0),
                        "importance_type": it.get("importance_type", kind),
                    })
        for key1, val1 in payload.items():
            if not isinstance(val1, dict):
                continue
            if kind in val1:
                add_items(val1.get(kind), "current", key1)
            else:
                for variant, val2 in val1.items():
                    if isinstance(val2, dict):
                        add_items(val2.get(kind), key1, variant)
    return pd.DataFrame(rows)


def get_best_per_target(df: pd.DataFrame, prefer_nested: bool = True) -> pd.DataFrame:
    if df.empty:
        return df
    d = ensure_result_columns(df)
    sort_col = "nested_cv_auroc_mean" if prefer_nested and "nested_cv_auroc_mean" in d.columns else "AUROC"
    if sort_col not in d.columns:
        sort_col = "AUROC"
    return d.sort_values(["target", sort_col, "AUPRC"], ascending=[True, False, False]).groupby("target", as_index=False).head(1).reset_index(drop=True)


def reconstruct_roc_curve(auroc: float, sensitivity: float, specificity: float) -> Tuple[np.ndarray, np.ndarray]:
    auroc = float(np.clip(safe_float(auroc, 0.5), 0.01, 0.99))
    sens = float(np.clip(safe_float(sensitivity, 0.5), 0.0, 1.0))
    spec = float(np.clip(safe_float(specificity, 0.5), 0.0, 1.0))
    fpr_op = 1.0 - spec
    tpr_op = sens
    fpr = np.linspace(0, 1, 200)
    gamma = max(0.05, 1.0 / max(auroc / (1 - auroc + 1e-6), 1e-6))
    tpr = np.power(fpr, gamma)
    # Blend with piecewise interpolation through the observed operating point.
    pw_fpr = np.array([0.0, fpr_op, 1.0])
    pw_tpr = np.array([0.0, tpr_op, 1.0])
    pw = np.interp(fpr, pw_fpr, pw_tpr)
    tpr = 0.65 * tpr + 0.35 * pw
    tpr = np.maximum.accumulate(np.clip(tpr, 0, 1))
    return fpr, tpr


def reconstruct_pr_curve(auprc: float, recall: float, precision: float, prevalence: float) -> Tuple[np.ndarray, np.ndarray]:
    auprc = float(np.clip(safe_float(auprc, prevalence), 0.01, 0.99))
    recall = float(np.clip(safe_float(recall, 0.5), 0, 1))
    precision = float(np.clip(safe_float(precision, prevalence), 0, 1))
    x = np.linspace(0.001, 1.0, 200)
    start = min(0.98, max(precision, auprc + 0.2, prevalence + 0.2))
    curve = prevalence + (start - prevalence) * np.power(1 - x, max(0.35, 1.5 * (1 - auprc)))
    pw = np.interp(x, [0.001, recall, 1.0], [start, precision, prevalence])
    y = 0.6 * curve + 0.4 * pw
    return x, np.clip(y, 0, 1)


def load_all_inputs(logger: logging.Logger) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    data["step03"] = pd.read_csv(INPUT_STEP03) if INPUT_STEP03.exists() else pd.DataFrame()
    data["step04"] = pd.read_csv(INPUT_STEP04) if INPUT_STEP04.exists() else pd.DataFrame()
    data["feature_dict"] = load_json_safe(INPUT_FEATURE_DICT)
    data["step04_summary"] = load_json_safe(first_existing(STEP04_SUMMARY_CANDIDATES))
    data["step05_summary"] = load_json_safe(first_existing(STEP05_SUMMARY_CANDIDATES))
    data["step05_results"] = ensure_result_columns(read_csv_first(STEP05_RESULTS_CANDIDATES))
    if data["step05_results"].empty and data["step05_summary"]:
        data["step05_results"] = summary_to_results_df(data["step05_summary"])
    data["step05_best"] = ensure_result_columns(read_csv_first(STEP05_BEST_CANDIDATES))
    if data["step05_best"].empty and not data["step05_results"].empty:
        data["step05_best"] = get_best_per_target(data["step05_results"])
    data["step05_variant"] = ensure_result_columns(read_csv_first(STEP05_VARIANT_CANDIDATES))
    data["step05_validation"] = ensure_result_columns(read_csv_first(STEP05_VALIDATION_CANDIDATES))
    # Restrict the importance figures to the prespecified radiomic arm. summary_to_importance_df
    # returns every feature-set variant, and taking the top-ranked rows across all of them put
    # PLS latent components from the acquisition_only model (latent_1..3) in the same column as
    # elastic-net absolute coefficients from the radiomic model, then rescaled both to [0,1] as
    # if they were one quantity. A PLS loading and an |coefficient| are not comparable, and the
    # rows were labelled as though they were all radiomic features.
    data["step05_native_importance"] = _restrict_to_variant(
        summary_to_importance_df(data["step05_summary"], "top_native_features"), IMPORTANCE_VARIANT)
    data["step05_perm_importance"] = _restrict_to_variant(
        summary_to_importance_df(data["step05_summary"], "top_permutation_features"), IMPORTANCE_VARIANT)
    data["step07_summary"] = load_json_safe(first_existing(STEP07_SUMMARY_CANDIDATES))
    data["step07_results"] = ensure_result_columns(read_csv_first(STEP07_RESULTS_CANDIDATES))
    data["step08_results"] = ensure_result_columns(read_csv_first(STEP08_RESULTS_CANDIDATES))
    data["step09_results"] = ensure_result_columns(read_csv_first(STEP09_RESULTS_CANDIDATES))
    data["step09_comparison"] = ensure_result_columns(read_csv_first(STEP09_COMPARISON_CANDIDATES))
    data["step05b_scores"] = read_csv_first(STEP05B_SCORE_CANDIDATES)
    # The prespecified radiomic arm, one row per target. Panels that show "the tabular model"
    # must use this, not step05_best: that file picks the winning feature-set variant per
    # target, which is acquisition_only -- the scanner control -- for ER and HER2.
    rad_rows = []
    for _t in TARGET_COLUMNS:
        _p = TABLES_DIR / ("step05master_%s_%s_best_model_test_result.csv" % (_t, IMPORTANCE_VARIANT))
        if _p.exists():
            _d = pd.read_csv(_p)
            if not _d.empty:
                rad_rows.append(_d.iloc[0])
    data["step05_radiomic"] = ensure_result_columns(
        pd.DataFrame(rad_rows).reset_index(drop=True) if rad_rows else pd.DataFrame())
    logger.info("Loaded Step03: %s", data["step03"].shape)
    logger.info("Loaded Step04: %s", data["step04"].shape)
    logger.info("Loaded Step05 results: %s", data["step05_results"].shape)
    logger.info("Loaded Step07 results: %s", data["step07_results"].shape)
    logger.info("Loaded Step08 results: %s", data["step08_results"].shape)
    logger.info("Loaded Step09 results: %s", data["step09_results"].shape)
    return data


def get_target_counts_by_split(step04: pd.DataFrame, target: str) -> pd.DataFrame:
    if step04.empty or target not in step04.columns or "split" not in step04.columns:
        return pd.DataFrame(columns=["split", "label", "count"])
    d = step04[["split", target]].dropna().copy()
    d[target] = d[target].astype(int)
    return d.groupby(["split", target], as_index=False).size().rename(columns={target: "label", "size": "count"})


def count_patients(step04: pd.DataFrame) -> int:
    if "patient_base" in step04.columns:
        return int(step04["patient_base"].nunique())
    return int(len(step04))


def count_cases(step03: pd.DataFrame) -> int:
    if "case_id" in step03.columns:
        return int(step03["case_id"].nunique())
    return 0


def _stage_auroc(comp: pd.DataFrame, target: str, stage: str) -> float:
    """AUROC for one (stage, target) in step09_final_comparison_all_stages.csv."""
    if comp is None or not isinstance(comp, pd.DataFrame) or comp.empty:
        return float("nan")
    if "stage" not in comp.columns or "target" not in comp.columns:
        return float("nan")
    col = next((c for c in ("AUROC", "Test_AUROC", "test_AUROC") if c in comp.columns), None)
    if col is None:
        return float("nan")
    sub = comp[(comp["stage"].astype(str) == stage) & (comp["target"].astype(str) == target)]
    return float("nan") if sub.empty else safe_float(sub.iloc[0][col], float("nan"))


def get_best_metric_row(data: Dict[str, Any], target: str, source: str = "step05") -> Optional[pd.Series]:
    if source == "step05_radiomic":
        df = data.get("step05_radiomic", pd.DataFrame())
        if df.empty:                      # fall back rather than draw nothing
            df = data.get("step05_best", pd.DataFrame())
    elif source == "step05":
        df = data.get("step05_best", pd.DataFrame())
        if df.empty:
            df = get_best_per_target(data.get("step05_results", pd.DataFrame()))
    elif source == "step07":
        df = data.get("step07_results", pd.DataFrame())
        if df.empty:
            return None
        # Select on VALIDATION only. "AUROC" (the locked-test value) must never appear in
        # these sort keys: HER2's two MIL variants tie exactly on valid_AUROC (0.5859), so a
        # test-based tiebreak silently picked the model with the better TEST score
        # (GatedAttentionMIL, 0.569) over the validation-selected one (AttentionMIL, 0.514) --
        # test-set leakage into model selection, in a study about leakage. Tie-break on the
        # model name instead: deterministic, test-blind, and it reproduces step_09's choice.
        sort_cols = [c for c in ["valid_composite_score", "valid_AUROC"] if c in df.columns]
        if sort_cols:
            name_col = [c for c in ["model", "model_display", "target_name"] if c in df.columns][:1]
            df = df.sort_values(["target"] + sort_cols + name_col,
                                ascending=[True] + [False] * len(sort_cols) + [True] * len(name_col)
                                ).groupby("target", as_index=False).head(1)
    else:
        df = pd.DataFrame()
    if df.empty or "target" not in df.columns:
        return None
    sub = ensure_result_columns(df[df["target"] == target])
    if sub.empty:
        return None
    return sub.iloc[0]


# ═══════════════════════════════════════════════════════════════════════════════
# Figure functions
# ═══════════════════════════════════════════════════════════════════════════════

def fig_00_graphical_abstract(data: Dict[str, Any], output_dir: Path, logger: logging.Logger) -> Dict[str, Any]:
    step03, step04 = data["step03"], data["step04"]
    n_pat = count_patients(step04)
    n_lesions = len(step03)
    n_cases = count_cases(step03)
    n_features = data.get("step05_summary", {}).get("feature_set_sizes", {}).get("radiomics_pure", "642")
    best_df = get_best_per_target(data.get("step05_results", pd.DataFrame()))
    best_auc = safe_float(best_df["AUROC"].max() if not best_df.empty and "AUROC" in best_df else np.nan, 0.5)
    fig = plt.figure(figsize=(11.2, 5.6))
    ax = fig.add_subplot(111)
    ax.axis("off")
    # Pin one coordinate system: the boxes are placed in DATA coords while the ROC insets
    # used axes-fraction coords. Without explicit limits the axes autoscale to the boxes
    # only (y≈0.56–0.82), so anything in axes-fraction coords landed on top of them.
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    boxes = [
        ("MRI +\nSegmentation", "165 patients screened"),
        ("Lesion\nExtraction", f"n={n_lesions} lesions"),
        ("Radiomics", f"{n_features} features"),
        ("Patient\nAggregation", f"n={n_pat} patients | {n_cases} cases"),
        ("Receptor\nPrediction", f"ER / PR / HER2\nAUC up to {best_auc:.2f}"),
    ]
    xs = np.linspace(0.10, 0.90, len(boxes))
    for i, ((title, sub), x) in enumerate(zip(boxes, xs)):
        _fc = PALETTE[i % len(PALETTE)]
        rect = patches.FancyBboxPatch((x - 0.075, 0.66), 0.15, 0.19, boxstyle="round,pad=0.015",
                                      facecolor=_fc, edgecolor="black", linewidth=1.2, alpha=0.95)
        ax.add_patch(rect)
        # PALETTE cycles through light fills too — pick the higher-contrast text colour
        ax.text(x, 0.755, title, ha="center", va="center", fontsize=11.5,
                color=_contrast_text(_fc), fontweight="bold", linespacing=1.4)
        ax.text(x, 0.60, sub, ha="center", va="center", fontsize=8.5, color="#333333", linespacing=1.4)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.093, 0.755), xytext=(x + 0.093, 0.755),
                        arrowprops=dict(arrowstyle="->", lw=2, color="#333333"))
    # Mini ROC sketches
    for j, target in enumerate(TARGET_COLUMNS):
        row = get_best_metric_row(data, target, "step05_radiomic")
        auc = extract_metric(row, ["AUROC"], 0.5) if row is not None else 0.5
        # lower band, clear of the pipeline boxes (which end at y≈0.645)
        x0 = 0.19 + j * 0.28
        y0, w, h = 0.19, 0.16, 0.24
        ax.add_patch(patches.Rectangle((x0, y0), w, h, fill=False, edgecolor="#BBBBBB", lw=0.9))
        ax.plot([x0, x0 + w], [y0, y0 + h], "--", color="gray", lw=1)
        fpr, tpr = reconstruct_roc_curve(auc, extract_metric(row, ["Sensitivity"], 0.6) if row is not None else 0.6, extract_metric(row, ["Specificity"], 0.6) if row is not None else 0.6)
        ax.plot(x0 + fpr * w, y0 + tpr * h, color=TARGET_COLORS[target], lw=2)
        ax.text(x0 + w / 2, y0 - 0.055, f"{TARGET_DISPLAY[target]} AUC={auc:.2f}",
                ha="center", va="top", fontsize=9, fontweight="bold", color="#333333")
    ax.set_title("BCBM Radiogenomics Pipeline", fontsize=16, fontweight="bold", pad=12)
    path = save_figure(fig, output_dir, "fig_00_graphical_abstract", svg=True)
    return standard_result(True, path, section="Title / Graphical Abstract")


def _contrast_text(facecolor) -> str:
    """Pick the text colour (white or near-black) with the HIGHER WCAG contrast ratio
    against `facecolor`. Comparing both ratios — rather than thresholding luminance —
    matters for mid-tone fills (teal/salmon), where white can look acceptable yet scores
    below the 4.5 target while black scores well above it."""
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(facecolor)
    lin = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    lum = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
    ratio_white = 1.05 / (lum + 0.05)
    ratio_black = (lum + 0.05) / 0.05
    return "#FFFFFF" if ratio_white >= ratio_black else "#111111"


def fig_01_cohort_consort(data: Dict[str, Any], output_dir: Path, logger: logging.Logger) -> Dict[str, Any]:
    step03, step04 = data["step03"], data["step04"]
    n_analysis_lesions = len(step03)
    n_pat = count_patients(step04)
    n_cases = count_cases(step03)
    split_counts = step04.groupby("split")["patient_base"].nunique().to_dict() if "split" in step04.columns and "patient_base" in step04.columns else {}
    # Derive the exclusion breakdown from step_03's FINAL mask categories rather than
    # hardcoding it. The previous hardcoded dict mixed the INITIAL counts (it listed
    # "unclear = 11", which descriptor resolution later reclassified into lesion/manual)
    # with the final ones, so the itemisation summed to 912 against a stated total of 984.
    s3 = load_json_safe(METADATA_DIR / "step03_summary.json")
    final_cats = s3.get("final_mask_category_counts", {}) if isinstance(s3, dict) else {}
    total_lesions_raw = int(sum(final_cats.values())) if final_cats else 2825
    excluded = max(0, total_lesions_raw - n_analysis_lesions)
    _label = {"target": "target", "cavity_or_bed": "cavity/bed",
              "other_structure": "other structure", "manual_review_required": "manual review"}
    excl_parts = {_label.get(k, k): int(v) for k, v in final_cats.items() if k != "lesion"}
    # Lesion-category masks that still did not reach the analysis table (missing labels or
    # unusable features). Including them makes the itemisation reconcile with the total.
    n_lesion_cat = int(final_cats.get("lesion", 0))
    residual = max(0, n_lesion_cat - n_analysis_lesions)
    if residual:
        excl_parts["lesion, not eligible"] = residual
    fig, ax = plt.subplots(figsize=(10.5, 8.4))
    ax.axis("off")

    def box(x, y, w, h, txt, fc="#FFFFFF", ec="#333333", ls="-"):
        r = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", facecolor=fc,
                                   edgecolor=ec, linestyle=ls, linewidth=1.6)
        ax.add_patch(r)
        # auto-contrast so text stays legible on dark fills
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=11.5,
                fontweight="bold", color=_contrast_text(fc), linespacing=1.5)

    yvals = [0.84, 0.66, 0.47, 0.29, 0.10]
    box(0.25, yvals[0], 0.50, 0.09, "Total patients in dataset\n165", PALETTE[0], "black")
    box(0.25, yvals[1], 0.50, 0.09, f"Lesions with segmentation\n{total_lesions_raw}", PALETTE[1], "black")
    box(0.25, yvals[2], 0.50, 0.09, f"Analysis-eligible lesions\n{n_analysis_lesions}", PALETTE[2], "black")
    box(0.25, yvals[3], 0.50, 0.09, f"Unique patients: {n_pat}\nUnique cases: {n_cases}", PALETTE[3], "black")
    for y1, y2 in zip(yvals[:-1], yvals[1:]):
        ax.annotate("", xy=(0.50, y2 + 0.10), xytext=(0.50, y1), arrowprops=dict(arrowstyle="->", lw=1.8))
    # Split boxes
    splits = [("Train", split_counts.get("train", 0)), ("Tuning", split_counts.get("valid", 0)), ("Test", split_counts.get("test", 0))]
    for i, (name, cnt) in enumerate(splits):
        box(0.18 + i * 0.22, yvals[4], 0.17, 0.09, f"{name}\n{cnt} patients", "#F8F9FA")
    # Exclusion branch — stack the breakdown one item per line so it stays inside the box
    # NOTE: FancyBboxPatch(boxstyle="round,pad=0.02") inflates each box by 0.02 on every
    # side, so the main column (nominal 0.25–0.75) actually renders 0.23–0.77. Place the
    # exclusion box at 0.82 (renders from 0.80) to leave a real gap for the arrow.
    ex_x, ex_w, ex_h = 0.82, 0.205, 0.20
    ex_y = yvals[1] + 0.045 - ex_h / 2
    excl_lines = "\n".join(f"{k} = {v}" for k, v in excl_parts.items())
    r = patches.FancyBboxPatch((ex_x, ex_y), ex_w, ex_h, boxstyle="round,pad=0.02",
                               facecolor="#E0E0E0", edgecolor="#555555", linestyle="--", linewidth=1.4)
    ax.add_patch(r)
    ax.text(ex_x + ex_w / 2, ex_y + ex_h / 2,
            f"Excluded non-lesion\nstructures: {excluded}\n\n{excl_lines}",
            ha="center", va="center", fontsize=8.5, color="#333333")
    # arrow spans the visual gap: from the main box's padded edge (0.77) to the
    # exclusion box's padded edge (0.80)
    ax.annotate("", xy=(ex_x - 0.022, yvals[1] + 0.045), xytext=(0.772, yvals[1] + 0.045),
                arrowprops=dict(arrowstyle="->", lw=1.5, linestyle="--", color="#555555"))
    # Autoscale only sees the *nominal* rects; the round-box pad (0.02/side) renders beyond
    # them and would be clipped at the axes edge (patches clip by default). Set limits that
    # include the padded extents: x 0.16–1.045, y 0.08–0.95 for the drawn artists.
    ax.set_xlim(0.13, 1.08)
    ax.set_ylim(0.04, 0.99)
    ax.set_title("CONSORT-style cohort and lesion filtering flow", fontsize=15, fontweight="bold")
    path = save_figure(fig, output_dir, "fig_01_cohort_consort")
    return standard_result(True, path, section="Methods")


def fig_02_label_distribution(data: Dict[str, Any], output_dir: Path, logger: logging.Logger) -> Dict[str, Any]:
    step04 = data["step04"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    splits = ["train", "valid", "test"]
    for ax, target in zip(axes, TARGET_COLUMNS):
        counts = get_target_counts_by_split(step04, target)
        x = np.arange(len(splits))
        neg, pos = [], []
        for s in splits:
            sub = counts[counts["split"] == s]
            neg.append(int(sub[sub["label"] == 0]["count"].sum()))
            pos.append(int(sub[sub["label"] == 1]["count"].sum()))
        width = 0.35
        ax.bar(x - width / 2, neg, width, color=GRAY, label="Negative")
        ax.bar(x + width / 2, pos, width, color=TARGET_COLORS[target], label="Positive")
        for i, (n0, n1) in enumerate(zip(neg, pos)):
            total = n0 + n1
            rate = 100 * n1 / total if total else 0
            ax.text(i + width / 2, n1 + 0.7, f"{rate:.0f}%", ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(["Tuning" if s == "valid" else s.title() for s in splits])
        ax.set_title(f"{TARGET_DISPLAY[target]} Status")
        ax.set_xlabel("Split")
        ax.grid(axis="y", alpha=0.25)
        # Pie inset — placed mid-right (below the legend area) to avoid overlap
        total_pos = sum(pos); total_neg = sum(neg)
        inset = ax.inset_axes([0.70, 0.34, 0.26, 0.30])
        if total_pos + total_neg > 0:
            inset.pie([total_pos, total_neg], colors=[TARGET_COLORS[target], LIGHT_GRAY], startangle=90, autopct="%1.0f%%", textprops={"fontsize": 7})
        inset.set_title("Overall +", fontsize=7)
    axes[0].set_ylabel("Patient count")
    axes[-1].legend(loc="upper right")
    fig.suptitle("Receptor status distribution across training, tuning and test sets", fontweight="bold")
    path = save_figure(fig, output_dir, "fig_02_label_distribution")
    return standard_result(True, path, section="Methods")


def fig_03_scanner_audit(data: Dict[str, Any], output_dir: Path, logger: logging.Logger) -> Dict[str, Any]:
    step04 = data["step04"]
    results = ensure_result_columns(data.get("step05_results", pd.DataFrame()))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    field_col = None
    for c in ["dominant_field_strength_t", "magnetic_field_strength", "field_strength"]:
        if c in step04.columns:
            field_col = c; break
    labels, vals_15, vals_3 = [], [], []
    if field_col:
        for target in TARGET_COLUMNS:
            for lbl in [1, 0]:
                sub = step04[step04[target] == lbl] if target in step04.columns else pd.DataFrame()
                fs = pd.to_numeric(sub[field_col], errors="coerce").round(1)
                labels.append(f"{TARGET_DISPLAY[target]}{'+' if lbl else '-'}")
                vals_15.append(int((fs == 1.5).sum()))
                vals_3.append(int((fs == 3.0).sum()))
        x = np.arange(len(labels))
        ax.bar(x, vals_15, label="1.5T", color=PALETTE[3])
        ax.bar(x, vals_3, bottom=vals_15, label="3T", color=PALETTE[0])
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Patients")
        ax.set_title("A. Field strength by receptor group")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Field strength column unavailable", ha="center", va="center")
        ax.axis("off")
    ax2 = axes[1]
    # Prefer the repeated nested cross-validation (step 05b), which is the study's primary
    # estimate and what Table 2 reports; a single partition of these data moves the acquisition
    # control by up to 0.13 AUROC, so plotting it here contradicted the table.
    rep = PROJECT_ROOT / "reports" / "tables" / "step05b_repeated_nested_cv_primary_summary.csv"
    scanner = pd.DataFrame()
    repeated = False
    if rep.exists():
        try:
            g = pd.read_csv(rep)
            g = g[g["feature_set_variant"] == "acquisition_only"]
            if not g.empty:
                inv = {v: k for k, v in TARGET_DISPLAY.items()}
                scanner = pd.DataFrame({
                    "target": [inv.get(d, d) for d in g["target_display"]],
                    "nested_cv_auroc_mean": g["mean"].astype(float).values,
                    "nested_cv_auroc_std": g["sd"].astype(float).values})
                repeated = True
        except Exception as e:
            logging.getLogger(STEP_NAME).warning("  repeated-CV summary unreadable (%s); using step 05", e)
    if scanner.empty and not results.empty:
        scanner = results[results.get("feature_set_variant", "") == "acquisition_only"].copy()
    if not scanner.empty:
        # Plot the NESTED-CV mean, which is the study's primary metric and the quantity the
        # legend and Results text describe (ER 0.46, PR 0.56, HER2 0.50). This panel
        # previously plotted the locked-test "AUROC" column instead, which shows PR as the
        # LOWEST of the three (0.55 vs 0.56/0.56) and so contradicted its own caption.
        best = scanner.sort_values(["target", "nested_cv_auroc_mean"], ascending=[True, False]).groupby("target", as_index=False).head(1)
        best = best[best["target"].isin(TARGET_COLUMNS)]
        y = np.arange(len(best))
        vals = best["nested_cv_auroc_mean"].astype(float).values
        sds = best["nested_cv_auroc_std"].astype(float).values if "nested_cv_auroc_std" in best.columns else np.zeros_like(vals)
        ax2.barh(y, vals, color=[TARGET_COLORS.get(t, GRAY) for t in best["target"]],
                 xerr=sds, error_kw=dict(ecolor="#444444", lw=1.2, capsize=4))
        ax2.axvline(0.5, color="gray", linestyle="--")
        ax2.set_yticks(y); ax2.set_yticklabels([TARGET_DISPLAY.get(t, t) for t in best["target"]])
        ax2.set_xlim(0.3, 0.8)
        ax2.set_xlabel(("Acquisition-only AUROC over 10 partitions (mean ± SD)" if repeated
                        else "Acquisition-only nested-CV AUROC, single partition (mean ± SD)"))
        ax2.set_title("B. Acquisition-only control")
        for yi, v, sd in zip(y, vals, sds):
            ax2.text(v + sd + 0.012, yi, f"{v:.2f}", va="center")
    else:
        ax2.text(0.5, 0.5, "Acquisition-only results unavailable", ha="center", va="center")
        ax2.axis("off")
    fig.suptitle("Scanner confounding audit", fontweight="bold")
    path = save_figure(fig, output_dir, "fig_03_scanner_audit")
    return standard_result(True, path, section="Methods")


def fig_04_feature_importance_heatmap(data: Dict[str, Any], output_dir: Path, logger: logging.Logger) -> Dict[str, Any]:
    imp = data.get("step05_native_importance", pd.DataFrame())
    if imp.empty:
        return standard_result(False, reason="No native importance data", section="Radiomics Features")
    top_features: List[str] = []
    matrices: Dict[str, Dict[str, float]] = {}
    rank_map: Dict[Tuple[str, str], int] = {}
    for target in TARGET_COLUMNS:
        sub = imp[imp["target"] == target].sort_values("importance", ascending=False).head(15)
        matrices[target] = dict(zip(sub["feature"].astype(str), sub["importance"].astype(float)))
        for rank, feat in enumerate(sub["feature"].astype(str), start=1):
            rank_map[(target, feat)] = rank
            if feat not in top_features:
                top_features.append(feat)
    top_features = top_features[:20]
    mat = np.zeros((len(top_features), len(TARGET_COLUMNS)))
    for j, target in enumerate(TARGET_COLUMNS):
        vals = np.array([matrices.get(target, {}).get(f, 0.0) for f in top_features], dtype=float)
        maxv = np.nanmax(vals) if vals.size else 1.0
        mat[:, j] = vals / max(maxv, 1e-9)
    fig, ax = plt.subplots(figsize=(9.0, max(5, 0.38 * len(top_features) + 2)))
    # Sequential light->dark: zero importance renders white (not dark), so the
    # significance markers and the few important cells stand out clearly.
    im = ax.imshow(mat, aspect="auto", cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(len(TARGET_COLUMNS)))
    ax.set_xticklabels([TARGET_DISPLAY[t] for t in TARGET_COLUMNS], fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels([clean_feature_name(f) for f in top_features], fontsize=8.5)
    # thin white gridlines to separate cells
    ax.set_xticks(np.arange(-0.5, len(TARGET_COLUMNS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(top_features), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    for i, feat in enumerate(top_features):
        for j, target in enumerate(TARGET_COLUMNS):
            rank = rank_map.get((target, feat), 999)
            # These marks are rank within the target's own ranking, not a significance test.
            # A sparse model leaves most coefficients at exactly zero, so ranks 2-5 can be a
            # tie at 0.0; marking those as top-ranked put bold stars on blank cells.
            star = "" if mat[i, j] < 1e-6 else ("***" if rank <= 5 else "*" if rank <= 10 else "")
            if star:
                ax.text(j, i, star, ha="center", va="center", fontsize=11, fontweight="bold",
                        color="#FFFFFF" if mat[i, j] > 0.55 else "#111111")
    # Family color strip
    fam_ax = ax.inset_axes([1.03, 0, 0.035, 1], transform=ax.transAxes)
    fam_arr = np.arange(len(top_features)).reshape(-1, 1)
    fam_colors = [FEATURE_FAMILY_COLORS.get(get_feature_family(f), GRAY) for f in top_features]
    for i, col in enumerate(fam_colors):
        fam_ax.add_patch(patches.Rectangle((0, i - 0.5), 1, 1, color=col))
    fam_ax.set_xlim(0, 1); fam_ax.set_ylim(len(top_features) - 0.5, -0.5); fam_ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.08)
    cbar.set_label("Normalized importance")
    # The family colour strip had no key, so its colours were unreadable. Add one listing only
    # the families actually present in this panel.
    fams_present, seen = [], set()
    for f in top_features:
        fam = get_feature_family(f)
        if fam not in seen:
            seen.add(fam); fams_present.append(fam)
    fam_handles = [patches.Patch(facecolor=FEATURE_FAMILY_COLORS.get(fam, GRAY), edgecolor="none", label=fam)
                   for fam in fams_present]
    ax.legend(handles=fam_handles, title="Feature family", loc="upper left",
              bbox_to_anchor=(1.10, 1.0), fontsize=8, title_fontsize=8, frameon=False)
    ax.set_title("Top-ranked features of the radiomic block (642 features), by receptor",
                 fontweight="bold")
    path = save_figure(fig, output_dir, "fig_04_feature_importance_heatmap")
    return standard_result(True, path, section="Radiomics Features")


def fig_05_radiomics_violin(data: Dict[str, Any], output_dir: Path, logger: logging.Logger) -> Dict[str, Any]:
    if not SEABORN_AVAILABLE:
        return standard_result(False, reason="seaborn unavailable", section="Radiomics Features")
    step04 = data["step04"]
    imp = data.get("step05_native_importance", pd.DataFrame())
    if step04.empty or imp.empty:
        return standard_result(False, reason="missing input", section="Radiomics Features")
    # Two panels per receptor rather than "the first six found": taking three per target and
    # breaking at six meant ER and PR filled the figure and HER2 never appeared, in a panel
    # captioned "by receptor". Only features with a non-zero coefficient qualify -- a sparse
    # model leaves most at exactly zero, and a zero-coefficient feature is not "top-ranked".
    PER_TARGET = 2
    feats: List[Tuple[str, str]] = []
    for target in TARGET_COLUMNS:
        sub = (imp[(imp["target"] == target) & (imp["importance"].astype(float) > 0)]
               .sort_values("importance", ascending=False))
        taken = 0
        for feat in sub["feature"].astype(str):
            if taken >= PER_TARGET:
                break
            if feat in step04.columns and (feat, target) not in feats:
                feats.append((feat, target)); taken += 1
    if not feats:
        return standard_result(False, reason="top features not found in patient table", section="Radiomics Features")
    n = len(feats)
    fig, axes = plt.subplots(1, n, figsize=(3.7 * n, 5.6), sharey=False, constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, (feat, target) in zip(axes, feats):
        d = step04[[feat, target]].dropna().copy()
        d["status"] = d[target].astype(int).map({0: "Negative", 1: "Positive"})
        vals = pd.to_numeric(d[feat], errors="coerce")
        d = d[vals.notna()].copy(); vals = vals[vals.notna()]
        if vals.std() > 0:
            d["value"] = (vals - vals.mean()) / vals.std()
        else:
            d["value"] = 0.0
        sns.violinplot(data=d, x="status", y="value", hue="status", order=["Negative", "Positive"], hue_order=["Negative", "Positive"], palette={"Positive": TARGET_COLORS[target], "Negative": GRAY}, ax=ax, inner=None, cut=0, legend=False)
        sns.stripplot(data=d, x="status", y="value", order=["Negative", "Positive"], color="black", alpha=0.35, size=3, jitter=0.18, ax=ax)
        ptxt = "p=NA"
        if SCIPY_AVAILABLE:
            pos = d[d["status"] == "Positive"]["value"]
            neg = d[d["status"] == "Negative"]["value"]
            if len(pos) > 1 and len(neg) > 1:
                try:
                    _, p = mannwhitneyu(pos, neg, alternative="two-sided")
                    ptxt = f"p={p:.3g}"
                except Exception:
                    pass
        wrapped = textwrap.fill(clean_feature_name(feat), width=20)
        ax.set_title(f"{TARGET_DISPLAY[target]}\n{wrapped}", fontsize=8.5)
        ax.set_xlabel("")
        ax.set_ylabel("Standardized value" if ax is axes[0] else "")
        ax.text(0.5, 0.99, ptxt, transform=ax.transAxes, ha="center", va="top", fontsize=8)
    fig.suptitle("Top-ranked radiomic-block features by receptor status (standardized; two per receptor)", fontweight="bold")
    path = save_figure(fig, output_dir, "fig_05_radiomics_violin")
    return standard_result(True, path, section="Radiomics Features")


def _crossfitted_scores(data: Dict[str, Any], target: str) -> Optional[pd.DataFrame]:
    """Per-patient cross-fitted out-of-fold scores for the radiomic block, or None."""
    df = data.get("step05b_scores", pd.DataFrame())
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    if not {"y_true", "score"}.issubset(df.columns):
        return None
    # step_05b labels its targets with the binarized column name (ER_bin), not step_13's
    # target_er, so match on the display name and fall back to the raw column.
    disp = TARGET_DISPLAY.get(target, target)
    if "target_display" in df.columns:
        sub = df[df["target_display"].astype(str) == disp]
    elif "target" in df.columns:
        sub = df[df["target"].astype(str) == target]
    else:
        return None
    if "feature_set_variant" in sub.columns:
        rad = sub[sub["feature_set_variant"].astype(str) == IMPORTANCE_VARIANT]
        if not rad.empty:
            sub = rad
    sub = sub.dropna(subset=["y_true", "score"])
    if sub.empty or sub["y_true"].nunique() < 2:
        return None
    return sub


def fig_06_roc_curves(data: Dict[str, Any], output_dir: Path, logger: logging.Logger) -> Dict[str, Any]:
    """Real ROC curves from the cross-fitted out-of-fold scores.

    The earlier version of this figure called reconstruct_roc_curve(auc, sensitivity,
    specificity) on the locked-test row: that invents a curve shape with the reported area
    passing through one operating point, on 16-17 patients. It was captioned "approximate",
    but a reader cannot tell an invented curve from a measured one by looking at it. This
    version plots the empirical curve of the scores themselves and draws nothing if they
    are unavailable.
    """
    from sklearn.metrics import roc_curve as _roc_curve, roc_auc_score
    missing = [t for t in TARGET_COLUMNS if _crossfitted_scores(data, t) is None]
    if len(missing) == len(TARGET_COLUMNS):
        return standard_result(False, section="Model Performance",
                               reason="cross-fitted out-of-fold scores unavailable "
                                      "(run step_05b to write step05b_crossfitted_scores_*.csv); "
                                      "refusing to draw reconstructed curves")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), sharex=True, sharey=True)
    for ax, target in zip(axes, TARGET_COLUMNS):
        color = TARGET_COLORS[target]
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Chance")
        sub = _crossfitted_scores(data, target)
        if sub is not None:
            y = sub["y_true"].astype(int).values
            p = sub["score"].astype(float).values
            fpr, tpr, _ = _roc_curve(y, p)
            auc = roc_auc_score(y, p)
            ax.plot(fpr, tpr, color=color, lw=2.5, drawstyle="steps-post",
                    label=f"Cross-fitted AUROC={auc:.2f}")
            ax.fill_between(fpr, 0, tpr, color=color, alpha=0.10, step="post")
            ax.text(0.04, 0.92, f"n={len(y)} ({int(y.sum())}+/{int((1-y).sum())}\u2212)",
                    transform=ax.transAxes, fontsize=8.5, color="#333333")
        else:
            ax.text(0.5, 0.5, "scores unavailable", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="#888888")
        ax.set_title(TARGET_DISPLAY[target])
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("ROC curves of the cross-fitted out-of-fold scores (radiomic block, 642 features)",
                 fontweight="bold")
    path = save_figure(fig, output_dir, "fig_06_roc_curves")
    return standard_result(True, path, section="Model Performance")


def fig_07_pr_curves(data: Dict[str, Any], output_dir: Path, logger: logging.Logger) -> Dict[str, Any]:
    """Real precision-recall curves from the same cross-fitted scores. See fig_06 for why the
    reconstructed version was removed."""
    from sklearn.metrics import precision_recall_curve as _pr_curve, average_precision_score
    missing = [t for t in TARGET_COLUMNS if _crossfitted_scores(data, t) is None]
    if len(missing) == len(TARGET_COLUMNS):
        return standard_result(False, section="Model Performance",
                               reason="cross-fitted out-of-fold scores unavailable "
                                      "(run step_05b); refusing to draw reconstructed curves")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), sharex=True, sharey=True)
    for ax, target in zip(axes, TARGET_COLUMNS):
        color = TARGET_COLORS[target]
        sub = _crossfitted_scores(data, target)
        if sub is not None:
            y = sub["y_true"].astype(int).values
            p = sub["score"].astype(float).values
            prevalence = float(y.mean())
            prec, rec, _ = _pr_curve(y, p)
            ap = average_precision_score(y, p)
            ax.axhline(prevalence, color="gray", linestyle="--", lw=1,
                       label=f"Prevalence={prevalence:.2f}")
            ax.plot(rec, prec, color=color, lw=2.5, label=f"Cross-fitted AP={ap:.2f}")
            # top-left: this panel puts its legend at the bottom left
            ax.text(0.04, 0.94, f"n={len(y)} ({int(y.sum())}+/{int((1-y).sum())}\u2212)",
                    transform=ax.transAxes, fontsize=8.5, color="#333333", va="top")
        else:
            ax.text(0.5, 0.5, "scores unavailable", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="#888888")
        ax.set_title(TARGET_DISPLAY[target])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower left", fontsize=8)
    fig.suptitle("Precision\u2013recall curves of the cross-fitted out-of-fold scores "
                 "(radiomic block, 642 features)", fontweight="bold")
    path = save_figure(fig, output_dir, "fig_07_pr_curves")
    return standard_result(True, path, section="Model Performance")


def fig_08_performance_summary_bar(data: Dict[str, Any], output_dir: Path, logger: logging.Logger) -> Dict[str, Any]:
    metrics = ["AUROC", "AUPRC", "Balanced_Acc", "F1"]
    fig, ax = plt.subplots(figsize=(14, 6))
    positions, labels, vals, colors, hatches = [], [], [], [], []
    x = 0
    for target in TARGET_COLUMNS:
        row = get_best_metric_row(data, target, "step05_radiomic")
        mil = get_best_metric_row(data, target, "step07")
        for m in metrics:
            positions.append(x); labels.append(f"{TARGET_DISPLAY[target]}\n{m}"); vals.append(extract_metric(row, [m], np.nan) if row is not None else np.nan); colors.append(TARGET_COLORS[target]); hatches.append(None); x += 1
            positions.append(x); labels.append(""); vals.append(extract_metric(mil, [m], np.nan) if mil is not None else np.nan); colors.append(TARGET_COLORS[target]); hatches.append("///"); x += 0.8
        x += 1
    bars = ax.bar(positions, vals, color=colors, edgecolor="black", alpha=0.85)
    for b, h in zip(bars, hatches):
        if h:
            b.set_hatch(h); b.set_alpha(0.45)
    ax.axhline(0.5, color="gray", linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.1)
    for p, v in zip(positions, vals):
        if not np.isnan(v):
            ax.text(p, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
    ax.text(0.5, -0.24, "Tabular = filled bars | MIL = hatched/semi-transparent bars", transform=ax.transAxes, ha="center", bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.35"))
    ax.set_title("Multi-metric model performance summary", fontweight="bold")
    path = save_figure(fig, output_dir, "fig_08_performance_summary_bar")
    return standard_result(True, path, section="Model Performance")


def fig_09_confusion_matrices(data: Dict[str, Any], output_dir: Path, logger: logging.Logger) -> Dict[str, Any]:
    if not SEABORN_AVAILABLE:
        return standard_result(False, reason="seaborn unavailable", section="Model Performance")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    for ax, target in zip(axes, TARGET_COLUMNS):
        row = get_best_metric_row(data, target, "step05_radiomic")
        tn = safe_int(row.get("TN") if row is not None else 0); fp = safe_int(row.get("FP") if row is not None else 0)
        fn = safe_int(row.get("FN") if row is not None else 0); tp = safe_int(row.get("TP") if row is not None else 0)
        mat = np.array([[tn, fp], [fn, tp]])
        total = mat.sum() if mat.sum() else 1
        annot = np.array([[f"{mat[i,j]}\n{100*mat[i,j]/total:.1f}%" for j in range(2)] for i in range(2)])
        sns.heatmap(mat, annot=annot, fmt="", cmap="Blues", cbar=False, ax=ax, xticklabels=["Pred -", "Pred +"], yticklabels=["True -", "True +"])
        sens = extract_metric(row, ["Sensitivity"], np.nan) if row is not None else np.nan
        spec = extract_metric(row, ["Specificity"], np.nan) if row is not None else np.nan
        ppv = extract_metric(row, ["PPV_Precision", "Precision"], np.nan) if row is not None else np.nan
        npv = extract_metric(row, ["NPV"], np.nan) if row is not None else np.nan
        # step_05 does not emit an NPV column, so this printed "NPV=nan" in all three
        # panels even though it is computable from the counts the panel already shows.
        if not np.isfinite(npv) and (tn + fn) > 0:
            npv = tn / (tn + fn)
        ax.set_title(f"{TARGET_DISPLAY[target]} (n={int(total)})")
        # PPV is TP/(TP+FP); with no positive prediction at all -- which is what the
        # PR model does on this test set -- that is 0/0, undefined rather than zero.
        # Upstream metric functions return 0.0 by convention, so print NA instead of
        # asserting a precision that was never estimated. Likewise NPV, sensitivity
        # and specificity when their denominators are empty.
        def _fmt(v, denom):
            return "NA" if denom == 0 or not np.isfinite(v) else f"{v:.2f}"
        ax.set_xlabel(f"Sens={_fmt(sens, tp + fn)} | Spec={_fmt(spec, tn + fp)}\n"
                      f"PPV={_fmt(ppv, tp + fp)} | NPV={_fmt(npv, tn + fn)}")
    mdl = ""
    r0 = get_best_metric_row(data, TARGET_COLUMNS[0], "step05_radiomic")
    if r0 is not None and "model_display" in r0.index:
        mdl = "  (per-receptor best model, radiomic block of 642 features)"
    fig.suptitle("Locked-test confusion matrices at the out-of-fold Youden threshold" + mdl,
                 fontweight="bold")
    path = save_figure(fig, output_dir, "fig_09_confusion_matrices")
    return standard_result(True, path, section="Model Performance")


def fig_10_model_comparison_heatmap(data: Dict[str, Any], output_dir: Path, logger: logging.Logger) -> Dict[str, Any]:
    if not SEABORN_AVAILABLE:
        return standard_result(False, reason="seaborn unavailable", section="Model Performance")
    val = data.get("step05_validation", pd.DataFrame())
    if val.empty:
        val = data.get("step05_results", pd.DataFrame())
    if val.empty:
        return standard_result(False, reason="validation/model table unavailable", section="Model Performance")
    d = ensure_result_columns(val)
    model_col = "model_display" if "model_display" in d.columns else "model"
    score_col = "mean_inner_cv_auroc" if "mean_inner_cv_auroc" in d.columns else "validation_AUROC_at_selection" if "validation_AUROC_at_selection" in d.columns else "AUROC"
    d["col"] = d["target"].map(TARGET_DISPLAY).fillna(d["target"].astype(str)) + "\n" + d.get("feature_set_variant", "variant").astype(str).map(variant_name)
    piv = d.pivot_table(index=model_col, columns="col", values=score_col, aggfunc="max")
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(piv.columns) + 5), max(5, 0.35 * len(piv) + 2)))
    sns.heatmap(piv, cmap="YlOrRd", vmin=0.4, vmax=0.8, annot=True, fmt=".2f", ax=ax, cbar_kws={"label": "AUROC"})
    ax.set_title("Model comparison heatmap across targets and feature variants", fontweight="bold")
    ax.set_xlabel("Target × variant")
    ax.set_ylabel("Model")
    path = save_figure(fig, output_dir, "fig_10_model_comparison_heatmap")
    return standard_result(True, path, section="Model Performance")


def fig_11_mil_attention_schematic(data: Dict[str, Any], output_dir: Path, logger: logging.Logger) -> Dict[str, Any]:
    step03 = data["step03"]
    if step03.empty or "patient_base" not in step03.columns:
        return standard_result(False, reason="step03 unavailable", section="MIL / Multi-lesion")
    counts = step03.groupby("patient_base").size().sort_values(ascending=False)
    patient = counts.index[0]
    n = int(min(counts.iloc[0], 30))
    # NOTE: these weights are illustrative (synthetic, seeded) — this figure is a
    # schematic of how MIL attention aggregates lesions, NOT real attention values
    # from a trained model. Labels below mark it as illustrative to avoid misreading.
    rng = np.random.default_rng(42)
    weights = rng.gamma(shape=1.2, scale=1.0, size=n); weights = weights / weights.sum()
    fig = plt.figure(figsize=(13, 5))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.25, 1])
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    x0 = 0.05; y = 0.55; w = 0.80 / n
    cmap = plt.cm.YlOrBr
    for i, wt in enumerate(weights):
        ax.add_patch(patches.Rectangle((x0 + i * w, y), w * 0.9, 0.16, facecolor=cmap(0.2 + 0.75 * wt / weights.max()), edgecolor="white"))
    ax.annotate("", xy=(0.92, y + 0.08), xytext=(0.87, y + 0.08), arrowprops=dict(arrowstyle="->", lw=2))
    ax.add_patch(patches.FancyBboxPatch((0.92, y - 0.02), 0.14, 0.20, boxstyle="round,pad=0.02", facecolor=PALETTE[1], edgecolor="black"))
    ax.text(0.99, y + 0.08, "Patient\nPrediction", ha="center", va="center", color=_contrast_text(PALETTE[1]), fontweight="bold")
    ax.text(0.45, 0.78, f"Patient {patient}: lesion instances represented as attention-weighted bags", ha="center", fontsize=11)
    ax.text(0.45, 0.04, "Schematic — illustrative weights, not trained-model attention", ha="center", fontsize=8, style="italic", color="gray")
    ax.set_xlim(0, 1.1); ax.set_ylim(0, 1)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(np.arange(1, n + 1), weights, color=[cmap(0.2 + 0.75 * wv / weights.max()) for wv in weights])
    ax2.set_xlabel("Lesion rank")
    ax2.set_ylabel("Attention weight")
    ax2.set_title("Attention distribution (illustrative)")
    fig.suptitle("Multi-Instance Learning: Lesion-Level Attention Aggregation (Schematic)", fontweight="bold")
    path = save_figure(fig, output_dir, "fig_11_mil_attention_schematic")
    return standard_result(True, path, section="MIL / Multi-lesion")


def fig_12_mil_vs_tabular(data: Dict[str, Any], output_dir: Path, logger: logging.Logger) -> Dict[str, Any]:
    fig, ax = plt.subplots(figsize=(9, 5.6))
    y = np.arange(len(TARGET_COLUMNS))
    # The tabular point must be the prespecified radiomic arm and must be the same number
    # Figure 4 plots. get_best_metric_row(..., "step05") returns the best feature-set variant
    # per target, which for ER and HER2 is acquisition_only -- the scanner control -- so this
    # panel used to compare MIL against the control while captioning it "tabular radiomics",
    # and disagreed with Figure 4 on every receptor. Both now read one collation.
    comp = data.get("step09_comparison", pd.DataFrame())
    for i, target in enumerate(TARGET_COLUMNS):
        tab_auc = _stage_auroc(comp, target, "Step05_Radiomics")
        mil = get_best_metric_row(data, target, "step07")
        mil_auc = extract_metric(mil, ["AUROC"], np.nan) if mil is not None else np.nan
        color = TARGET_COLORS[target]
        if not np.isnan(tab_auc) and not np.isnan(mil_auc):
            ax.plot([tab_auc, mil_auc], [i, i], color=color, lw=2, alpha=0.6)
        coincident = (not np.isnan(tab_auc) and not np.isnan(mil_auc)
                      and abs(tab_auc - mil_auc) < 5e-3)
        if not np.isnan(tab_auc):
            ax.scatter(tab_auc, i, s=90, color=color, edgecolor="black", label="Tabular" if i == 0 else None, zorder=3)
            if not coincident:
                ax.text(tab_auc, i + 0.22, f"{tab_auc:.2f}", ha="center", va="bottom", fontsize=9)
        if not np.isnan(mil_auc):
            ax.scatter(mil_auc, i, s=90, facecolor="white", edgecolor=color, marker="D", linewidth=2, label="MIL" if i == 0 else None, zorder=3)
            ax.text(mil_auc, i - 0.22, f"{mil_auc:.2f}", ha="center", va="top", fontsize=9)
        if coincident:
            # Equal AUROCs put the filled circle underneath the hollow diamond; without this the
            # panel shows one marker and prints the same number above and below it.
            ax.text(mil_auc, i + 0.22, f"{tab_auc:.2f} (both)", ha="center", va="bottom",
                    fontsize=9, fontstyle="italic")
    ax.axvline(0.5, color="gray", linestyle="--")
    ax.set_yticks(y); ax.set_yticklabels([TARGET_DISPLAY[t] for t in TARGET_COLUMNS])
    ax.set_xlim(0.35, 1.0)
    ax.set_ylim(-0.6, len(TARGET_COLUMNS) - 0.4)
    ax.set_xlabel("Locked-test AUROC (n = 16-17)")
    ax.set_title("Attention MIL versus tabular radiomics (radiomic block, 642 features)",
                 fontweight="bold", pad=12)
    ax.legend(loc="upper right")
    path = save_figure(fig, output_dir, "fig_12_mil_vs_tabular")
    return standard_result(True, path, section="MIL / Multi-lesion")


def fig_13_generalization_gap(data: Dict[str, Any], output_dir: Path, logger: logging.Logger) -> Dict[str, Any]:
    d = ensure_result_columns(data.get("step05_results", pd.DataFrame()))
    if d.empty:
        return standard_result(False, reason="step05 results unavailable", section="Discussion")
    fig, ax = plt.subplots(figsize=(8.6, 7))
    # Shading is deliberately NOT labelled "Generalization zone". The manuscript legend states
    # that points above the identity line reflect the instability of a 16-17-patient test set
    # rather than generalization, so calling that region "generalization" contradicted the text.
    # Neutral, descriptive labels are used instead.
    ax.fill_between([0, 1], [0, 1], [0, 0], color="#FDEDEC", alpha=0.6, zorder=0)
    ax.fill_between([0, 1], [1, 1], [0, 1], color="#EAF7EA", alpha=0.6, zorder=0)
    marker_map = {"radiomics_pure": "o", "radiomics_plus_burden": "D", "all_features": "s", "acquisition_only": "^"}
    # Block sizes come from the dictionary the analysis used. Typed in, they went stale when
    # four mask-derived columns moved from acquisition to lesion burden (25 -> 21, 717 -> 721).
    fd = data.get("feature_dict") or {}
    n_rad = len(fd.get("radiomic_block_columns", [])) or 642
    n_acq = len(fd.get("acquisition_block_columns", [])) or 21
    n_bur = len(fd.get("burden_study_block_columns", [])) or 79
    variant_label = {"radiomics_pure": "Radiomics only (%d)" % n_rad,
                     "radiomics_plus_burden": "+ burden/study (%d)" % (n_rad + n_bur),
                     "all_features": "All features (%d)" % (n_rad + n_bur + n_acq),
                     "acquisition_only": "Acquisition only (%d)" % n_acq}
    for _, r in d.iterrows():
        target = r.get("target")
        x = extract_metric(r, ["validation_AUROC_at_selection", "nested_cv_auroc_mean", "valid_AUROC"], np.nan)
        y = extract_metric(r, ["AUROC"], np.nan)
        if np.isnan(x) or np.isnan(y):
            continue
        variant = str(r.get("feature_set_variant", "radiomics_pure"))
        # Constant marker size: size previously encoded n_selected_features, which no key or
        # legend explained, so readers saw an unexplained third dimension.
        ax.scatter(x, y, s=110, color=TARGET_COLORS.get(target, GRAY), marker=marker_map.get(variant, "o"),
                   edgecolor="black", alpha=0.9, zorder=3)
        gap = x - y
        if abs(gap) > 0.2:
            ax.text(x + 0.005, y + 0.005, str(r.get("model_display", r.get("model", "")))[:16], fontsize=7)
    ax.plot([0, 1], [0, 1], "--", color="black", lw=1, zorder=2)
    # Keys for BOTH encodings — the figure previously had none, so a reader could not tell which
    # point was which receptor or which feature variant.
    from matplotlib.lines import Line2D
    tgt_handles = [Line2D([], [], marker="o", linestyle="", markersize=9,
                          markerfacecolor=TARGET_COLORS.get(t, GRAY), markeredgecolor="black",
                          label=TARGET_DISPLAY.get(t, t)) for t in TARGET_COLUMNS]
    var_handles = [Line2D([], [], marker=mk, linestyle="", markersize=9, markerfacecolor="white",
                          markeredgecolor="black", label=variant_label[v]) for v, mk in marker_map.items()]
    leg1 = ax.legend(handles=tgt_handles, title="Receptor", loc="upper left", fontsize=9, title_fontsize=9, framealpha=0.95)
    ax.add_artist(leg1)
    ax.legend(handles=var_handles, title="Feature variant", loc="lower right", fontsize=9, title_fontsize=9, framealpha=0.95)
    # placed clear of both legend boxes (upper-left and lower-right)
    ax.text(0.63, 0.95, "locked test > nested CV", fontsize=9, color="#3F6B3F", style="italic")
    ax.text(0.80, 0.52, "locked test < nested CV", fontsize=9, color="#9A544C", style="italic", ha="center")
    ax.set_xlim(0.3, 0.9); ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("Nested cross-validation AUROC (single partition, step 05)")
    ax.set_ylabel("Locked-test AUROC")
    ax.set_title("Generalization gap diagnostic", fontweight="bold")
    path = save_figure(fig, output_dir, "fig_13_generalization_gap")
    return standard_result(True, path, section="Discussion")


def fig_14_feature_stability(data: Dict[str, Any], output_dir: Path, logger: logging.Logger) -> Dict[str, Any]:
    native = data.get("step05_native_importance", pd.DataFrame())
    perm = data.get("step05_perm_importance", pd.DataFrame())
    if native.empty:
        return standard_result(False, reason="native importance unavailable", section="Discussion")
    fig, axes = plt.subplots(3, 1, figsize=(11, 13.5), constrained_layout=True)
    for ax, target in zip(axes, TARGET_COLUMNS):
        # De-duplicate by feature name first: the importance table aggregates across models
        # and variants, so the same feature otherwise appears two or three times in the
        # top-10 with identical bars.
        nsub = (native[native["target"] == target]
                .sort_values("importance", ascending=False)
                .drop_duplicates(subset="feature", keep="first")
                .head(10))
        feats = nsub["feature"].astype(str).tolist()
        pmap = {}
        if not perm.empty:
            psub = perm[perm["target"] == target]
            pmap = dict(zip(psub["feature"].astype(str), psub["importance"].astype(float)))
        nat_vals = np.array([safe_float(v, 0.0) for v in nsub["importance"]])
        per_vals = np.array([pmap.get(f, 0.0) for f in feats], dtype=float)
        nat_vals = nat_vals / max(nat_vals.max(), 1e-9)
        per_vals = per_vals / max(per_vals.max(), 1e-9) if per_vals.max() > 0 else per_vals
        y = np.arange(len(feats))
        cols = [FEATURE_FAMILY_COLORS.get(get_feature_family(f), GRAY) for f in feats]
        ax.barh(y - 0.18, nat_vals, height=0.32, color=cols, label="Native")
        ax.barh(y + 0.18, per_vals, height=0.32, color="white", edgecolor=cols, hatch="///", label="Permutation")
        for yi, f, pv in zip(y, feats, per_vals):
            if pv > 0:
                ax.text(1.02, yi, "★", va="center", fontsize=10)
        ax.set_yticks(y); ax.set_yticklabels([clean_feature_name(f) for f in feats], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.15)
        ax.set_title(TARGET_DISPLAY[target])
        ax.set_xlabel("Normalized importance")
        if ax is axes[0]: ax.legend(loc="lower right")
    fig.suptitle("Feature importance stability: native vs permutation", fontweight="bold")
    path = save_figure(fig, output_dir, "fig_14_feature_stability")
    return standard_result(True, path, section="Discussion")


def fig_15_cross_stage_comparison(data: Dict[str, Any], output_dir: Path, logger: logging.Logger) -> Dict[str, Any]:
    """Horizontal AUROC comparison: Tabular → MIL → CNN → Hybrid, one row per receptor target."""
    from matplotlib.lines import Line2D

    df = data.get("step09_comparison", pd.DataFrame())
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return standard_result(False, reason="cross-stage comparison data unavailable (run step_09 first)", section="Model Performance")

    for required in ("stage", "target"):
        if required not in df.columns:
            logger.warning("fig_15: '%s' column missing in comparison data — skipping.", required)
            return standard_result(False, reason=f"'{required}' column missing", section="Model Performance")

    auroc_col = next((c for c in ["AUROC", "Test_AUROC", "test_AUROC"] if c in df.columns), None)
    if auroc_col is None:
        return standard_result(False, reason="No AUROC column found in comparison data", section="Model Performance")

    STAGE_CONFIG = [
        (["Step05_Radiomics", "Step05_Tabular", "Tabular"], "Tabular  (Step 05)", "o"),
        (["Step08A_MIL", "Step07_MIL", "MIL"],             "MIL      (Step 07)", "D"),
        (["Step08B_CNN", "Step08_CNN", "CNN"],             "CNN      (Step 08)", "s"),
        # Not "Hybrid": Step09_Selected is whichever variant validation chose, and for ER and
        # HER2 that was image_only, not the fusion model. Table S1 lists all three variants.
        (["Step09_Selected", "Step09_Hybrid", "Step08C_Hybrid", "Hybrid"], "Validation-selected (Step 09)", "^"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axvline(0.5, color="gray", linestyle="--", lw=1.2, alpha=0.65, label="_nolegend_")

    for yi, target in enumerate(TARGET_COLUMNS):
        color = TARGET_COLORS[target]
        sub = df[df["target"] == target]

        # Collect (auroc, marker) in stage order, skip unavailable
        points: List[Tuple[float, str]] = []
        for stage_keys, _, marker in STAGE_CONFIG:
            stage_rows = sub[sub["stage"].astype(str).isin(stage_keys)]
            if stage_rows.empty:
                continue
            val = safe_float(stage_rows.iloc[0][auroc_col], np.nan)
            if not np.isnan(val):
                points.append((val, marker))

        if not points:
            continue

        # Connecting line through available points (in stage order)
        if len(points) > 1:
            ax.plot([p[0] for p in points], [yi] * len(points),
                    color=color, lw=2.2, alpha=0.40, zorder=1)

        # Markers + AUROC labels. Stages can land on near-identical AUROC
        # (eg hybrid == tabular), so stagger labels that would collide and
        # print a shared label once for exactly-coincident points.
        MIN_SEP = 0.035          # x-distance below which two labels overlap
        drawn: List[Tuple[float, float]] = []   # (x, y) of labels already placed
        placed_x: List[float] = []              # marker x positions already occupied
        for val, marker in points:
            # Stages can land on exactly the same AUROC (HER2: tabular and validation-selected
            # are both 0.500). Drawn at the same y they hide each other, and the row appears to
            # have fewer stages than it does. Offset the later one vertically instead.
            n_same = sum(1 for x in placed_x if abs(val - x) < 1e-9)
            y_off = 0.075 * n_same
            placed_x.append(val)
            ax.scatter(val, yi + y_off, s=120, color=color, marker=marker,
                       edgecolor="black", linewidth=1.2, zorder=3)
            if n_same:
                continue                        # identical value already labelled
            dy = 0.14
            while any(abs(val - x) < MIN_SEP and abs(dy - y) < 0.12 for x, y in drawn):
                dy += 0.17                      # push this label to a free row
            ax.text(val, yi + dy, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=8,
                    color=color, fontweight="bold")
            drawn.append((val, dy))

    ax.set_yticks(range(len(TARGET_COLUMNS)))
    ax.set_yticklabels([TARGET_DISPLAY[t] for t in TARGET_COLUMNS], fontsize=13)
    ax.set_ylim(-0.65, len(TARGET_COLUMNS) - 0.10)   # headroom for staggered labels
    ax.set_xlim(0.25, 1.02)
    ax.set_xlabel("Locked-test AUROC", fontsize=12)
    ax.set_title("Cross-Stage Model Comparison\nTabular → MIL → CNN → validation-selected step-09 variant",
                 fontweight="bold", fontsize=14)
    ax.grid(axis="x", alpha=0.22)
    ax.text(0.5, -0.055, "Random", transform=ax.get_xaxis_transform(),
            ha="center", fontsize=8, color="gray")

    # Legend: stage markers (gray fill so all targets show equally)
    stage_handles = [
        Line2D([0], [0], marker=mk, color="w", markerfacecolor=GRAY,
               markeredgecolor="black", markersize=10, label=lbl)
        for _, lbl, mk in STAGE_CONFIG
    ]
    # Legend: target colors
    target_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=TARGET_COLORS[t], markeredgecolor="black",
               markersize=10, label=TARGET_DISPLAY[t])
        for t in TARGET_COLUMNS
    ]
    leg_stage = ax.legend(handles=stage_handles, loc="lower right",
                          fontsize=9, title="Stage", framealpha=0.9)
    ax.add_artist(leg_stage)
    ax.legend(handles=target_handles, loc="upper left",
              fontsize=9, title="Receptor", framealpha=0.9)

    fig.tight_layout()
    path = save_figure(fig, output_dir, "fig_15_cross_stage_comparison")
    return standard_result(True, path, section="Model Performance")


def fig_S1_lesion_category_breakdown(data: Dict[str, Any], output_dir: Path, logger: logging.Logger) -> Dict[str, Any]:
    step03 = data["step03"]
    lesion_count = len(step03)
    # Derive from step_03's FINAL mask categories rather than hardcoding. The remainder is
    # NOT "unclear": that category (11) existed only in the INITIAL categorisation and was
    # resolved away. The residual is lesion-category masks that did not reach the analysis
    # table (missing labels or unusable features).
    s3 = load_json_safe(METADATA_DIR / "step03_summary.json")
    final_cats = s3.get("final_mask_category_counts", {}) if isinstance(s3, dict) else {}
    total_raw = int(sum(final_cats.values())) if final_cats else 2825
    excluded_total = max(0, total_raw - lesion_count)
    _lbl = {"target": "Target", "other_structure": "Other", "cavity_or_bed": "Cavity/Bed",
            "manual_review_required": "Manual review"}
    excl = {_lbl.get(k, k): int(v) for k, v in final_cats.items() if k != "lesion"}
    residual = max(0, int(final_cats.get("lesion", 0)) - lesion_count)
    if residual:
        excl["Lesion, not eligible"] = residual
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.pie([lesion_count, excluded_total], radius=1.0, labels=[f"Lesion\n{lesion_count}", f"Excluded\n{excluded_total}"], colors=["#4CAF50", "#BDBDBD"], startangle=90, wedgeprops=dict(width=0.30, edgecolor="white"), autopct="%1.1f%%", pctdistance=0.85)
    inner_colors = ["#757575", "#9E9E9E", "#BDBDBD", "#D0D0D0", "#E0E0E0"]
    inner_wedges, _ = ax.pie(list(excl.values()), radius=0.68, colors=inner_colors, startangle=90,
                             wedgeprops=dict(width=0.28, edgecolor="white"))
    # Small inner slices overlap if labelled in-place; use a side legend instead.
    ax.legend(inner_wedges, [f"{k}: {v}" for k, v in excl.items()],
              title="Excluded breakdown", loc="center left",
              bbox_to_anchor=(1.0, 0.5), fontsize=9, frameon=False)
    ax.set_title("Segmentation category breakdown", fontweight="bold")
    path = save_figure(fig, output_dir, "fig_S1_lesion_category_breakdown")
    return standard_result(True, path, section="Supplementary")


def fig_S2_patient_lesion_burden(data: Dict[str, Any], output_dir: Path, logger: logging.Logger) -> Dict[str, Any]:
    step03, step04 = data["step03"], data["step04"]
    if step03.empty or "patient_base" not in step03.columns:
        return standard_result(False, reason="step03 unavailable", section="Supplementary")
    counts = step03.groupby("patient_base").size().rename("n_lesions").reset_index()
    d = counts.merge(step04[["patient_base"] + [t for t in TARGET_COLUMNS if t in step04.columns]], on="patient_base", how="left") if "patient_base" in step04.columns else counts
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(counts["n_lesions"], bins=30, color=PALETTE[0], alpha=0.85, edgecolor="white")
    if SCIPY_AVAILABLE and len(counts) > 3:
        xs = np.linspace(counts["n_lesions"].min(), counts["n_lesions"].max(), 200)
        try:
            kde = gaussian_kde(counts["n_lesions"])
            axes[0].plot(xs, kde(xs) * len(counts) * (xs[1] - xs[0]) * 30, color=PALETTE[2], lw=2)
        except Exception:
            pass
    axes[0].set_title("A. Lesion burden per patient")
    axes[0].set_xlabel("Number of lesions")
    axes[0].set_ylabel("Patients")
    axes[0].text(0.95, 0.95, f"Mean={counts['n_lesions'].mean():.1f}\nMedian={counts['n_lesions'].median():.1f}\nMax={counts['n_lesions'].max()}", transform=axes[0].transAxes, ha="right", va="top", bbox=dict(facecolor="white", alpha=0.85))
    if SEABORN_AVAILABLE:
        long_rows = []
        for target in TARGET_COLUMNS:
            if target in d.columns:
                for _, r in d[["n_lesions", target]].dropna().iterrows():
                    long_rows.append({"Target": TARGET_DISPLAY[target], "Status": "Positive" if int(r[target]) == 1 else "Negative", "n_lesions": r["n_lesions"]})
        long = pd.DataFrame(long_rows)
        if not long.empty:
            sns.boxplot(data=long, x="Target", y="n_lesions", hue="Status", ax=axes[1], palette={"Positive": PALETTE[2], "Negative": GRAY})
            axes[1].set_title("B. Lesion burden by receptor status")
            axes[1].set_ylabel("Number of lesions")  # was the raw column name "n_lesions"
        else:
            axes[1].axis("off")
    else:
        axes[1].axis("off")
    fig.suptitle("Patient lesion burden distribution", fontweight="bold")
    path = save_figure(fig, output_dir, "fig_S2_patient_lesion_burden")
    return standard_result(True, path, section="Supplementary")


def fig_S3_field_strength_split_balance(data: Dict[str, Any], output_dir: Path, logger: logging.Logger) -> Dict[str, Any]:
    step04 = data["step04"]
    if step04.empty or "split" not in step04.columns:
        return standard_result(False, reason="step04 split unavailable", section="Supplementary")
    field_col = next((c for c in ["dominant_field_strength_t", "magnetic_field_strength", "field_strength"] if c in step04.columns), None)
    if not field_col:
        return standard_result(False, reason="field strength unavailable", section="Supplementary")
    splits = ["train", "valid", "test"]
    vals15, vals3 = [], []
    for s in splits:
        fs = pd.to_numeric(step04[step04["split"] == s][field_col], errors="coerce").round(1)
        vals15.append(int((fs == 1.5).sum())); vals3.append(int((fs == 3.0).sum()))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(splits)); width = 0.35
    ax.bar(x - width/2, vals15, width, label="1.5T", color=PALETTE[3])
    ax.bar(x + width/2, vals3, width, label="3T", color=PALETTE[0])
    ax.set_xticks(x); ax.set_xticklabels(["Tuning" if s == "valid" else s.title() for s in splits])
    ax.set_ylabel("Patient count")
    ax2 = ax.twinx()
    for target in TARGET_COLUMNS:
        if target in step04.columns:
            rates = [step04[step04["split"] == s][target].dropna().mean() for s in splits]
            ax2.plot(x, rates, marker="o", color=TARGET_COLORS[target], label=f"{TARGET_DISPLAY[target]}+ rate")
    ax2.set_ylabel("Positive rate")
    ax2.set_ylim(0, 1)
    ax.legend(loc="upper left"); ax2.legend(loc="upper right", fontsize=8)
    ax.set_title("Field strength and receptor balance across splits", fontweight="bold")
    path = save_figure(fig, output_dir, "fig_S3_field_strength_split_balance")
    return standard_result(True, path, section="Supplementary")


def fig_S4_feature_missingness(data: Dict[str, Any], output_dir: Path, logger: logging.Logger) -> Dict[str, Any]:
    step04 = data["step04"]
    if step04.empty:
        return standard_result(False, reason="step04 unavailable", section="Supplementary")
    numeric_cols = [c for c in step04.columns if pd.api.types.is_numeric_dtype(step04[c]) and c not in TARGET_COLUMNS]
    miss = step04[numeric_cols].isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0].head(40)
    if miss.empty:
        return standard_result(False, reason="no missing numeric features", section="Supplementary")
    fig, ax = plt.subplots(figsize=(10, max(5, 0.25 * len(miss) + 2)))
    cols = [FEATURE_FAMILY_COLORS.get(get_feature_family(c), GRAY) for c in miss.index]
    y = np.arange(len(miss))
    ax.barh(y, miss.values * 100, color=cols)
    ax.set_yticks(y); ax.set_yticklabels([clean_feature_name(c) for c in miss.index], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(40, color="red", linestyle="--", lw=1, label="40% threshold")
    ax.set_xlabel("Missingness (%)")
    ax.set_title("Feature missingness profile", fontweight="bold")
    ax.legend()
    path = save_figure(fig, output_dir, "fig_S4_feature_missingness")
    return standard_result(True, path, section="Supplementary")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    global PROJECT_ROOT, DATA_DIR, PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR, TABLES_DIR, METADATA_DIR, LOGS_DIR, CHECKPOINTS_DIR, OUTPUT_DIR
    _ap = argparse.ArgumentParser(add_help=False)
    _ap.add_argument("project_root_pos", nargs="?", default=None, help="Optional project root path")
    _ap.add_argument("--force",    action="store_true")
    _ap.add_argument("--no-cache", action="store_true")
    _flags, _ = _ap.parse_known_args()

    if _flags.project_root_pos:
        PROJECT_ROOT = Path(_flags.project_root_pos).resolve()
        DATA_DIR = PROJECT_ROOT / "data"
        PROCESSED_DIR = DATA_DIR / "processed"
        REPORTS_DIR = PROJECT_ROOT / "reports"
        FIGURES_DIR = REPORTS_DIR / "figures"
        TABLES_DIR = REPORTS_DIR / "tables"
        METADATA_DIR = PROJECT_ROOT / "metadata"
        LOGS_DIR = PROJECT_ROOT / "logs"
        CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
        OUTPUT_DIR = FIGURES_DIR / "step13_publication_package"

    setup_global_style()
    cfg = Step10Config(project_root=str(PROJECT_ROOT), output_dir=str(OUTPUT_DIR))
    set_seed(cfg.global_seed)
    for p in [REPORTS_DIR, FIGURES_DIR, TABLES_DIR, METADATA_DIR, LOGS_DIR, CHECKPOINTS_DIR, OUTPUT_DIR]:
        ensure_dir(p)

    logger = setup_logger(LOGS_DIR / f"{STEP_NAME}.log")
    state = StateManager(CHECKPOINTS_DIR / "pipeline_state.json")
    logger.info("Starting %s", STEP_NAME)

    # ── Cache check ───────────────────────────────────────────────────────────
    _manifest = PROJECT_ROOT / "cache" / "step13_cache_manifest.json"
    _key_inputs = [p for p in [
        first_existing(STEP05_RESULTS_CANDIDATES),
        first_existing(STEP09_COMPARISON_CANDIDATES),
        first_existing(STEP07_RESULTS_CANDIDATES),
    ] if p is not None]
    _outputs = [OUTPUT_DIR / "figure_manifest.csv", METADATA_DIR / "step13_publication_summary.json"]
    if not _flags.force and not _flags.no_cache and _is_step_cached(_manifest, _key_inputs, "v1", _outputs):
        logger.info("Cache valid — Step 13 outputs unchanged. Skipping (use --force to re-run).")
        return

    data = load_all_inputs(logger)

    figures: List[Tuple[str, Callable[[Dict[str, Any], Path, logging.Logger], Dict[str, Any]]]] = [
        ("fig_00_graphical_abstract", fig_00_graphical_abstract),
        ("fig_01_cohort_consort", fig_01_cohort_consort),
        ("fig_02_label_distribution", fig_02_label_distribution),
        ("fig_03_scanner_audit", fig_03_scanner_audit),
        ("fig_04_feature_importance_heatmap", fig_04_feature_importance_heatmap),
        ("fig_05_radiomics_violin", fig_05_radiomics_violin),
        ("fig_06_roc_curves", fig_06_roc_curves),
        ("fig_07_pr_curves", fig_07_pr_curves),
        ("fig_08_performance_summary_bar", fig_08_performance_summary_bar),
        ("fig_09_confusion_matrices", fig_09_confusion_matrices),
        ("fig_10_model_comparison_heatmap", fig_10_model_comparison_heatmap),
        ("fig_11_mil_attention_schematic", fig_11_mil_attention_schematic),
        ("fig_12_mil_vs_tabular", fig_12_mil_vs_tabular),
        ("fig_13_generalization_gap", fig_13_generalization_gap),
        ("fig_14_feature_stability", fig_14_feature_stability),
        ("fig_15_cross_stage_comparison", fig_15_cross_stage_comparison),
        ("fig_S1_lesion_category_breakdown", fig_S1_lesion_category_breakdown),
        ("fig_S2_patient_lesion_burden", fig_S2_patient_lesion_burden),
        ("fig_S3_field_strength_split_balance", fig_S3_field_strength_split_balance),
        ("fig_S4_feature_missingness", fig_S4_feature_missingness),
    ]

    results: List[Dict[str, Any]] = []
    artifacts: Dict[str, str] = {"step10_log_file": str(LOGS_DIR / f"{STEP_NAME}.log")}
    for name, fn in figures:
        res = run_figure(fn, data, OUTPUT_DIR, logger, name)
        row = {"figure": name, **res}
        results.append(row)
        if res.get("success") and res.get("output_path"):
            artifacts[f"step10_{name}"] = str(res["output_path"])

    manifest = pd.DataFrame(results)
    manifest_path = OUTPUT_DIR / "figure_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    artifacts["step10_figure_manifest_csv"] = str(manifest_path)

    summary = {
        "config": asdict(cfg),
        "input_shapes": {
            "step03": list(data.get("step03", pd.DataFrame()).shape),
            "step04": list(data.get("step04", pd.DataFrame()).shape),
            "step05_results": list(data.get("step05_results", pd.DataFrame()).shape),
            "step07_results": list(data.get("step07_results", pd.DataFrame()).shape),
            "step08_results": list(data.get("step08_results", pd.DataFrame()).shape),
            "step09_results": list(data.get("step09_results", pd.DataFrame()).shape),
        },
        "figures": results,
        "notes": {
            "integration": "Publication figure suite produced by Step 13 (reporting).",
            "probability_note": "ROC/PR curves are approximated from stored operating-point metrics when raw probabilities are unavailable.",
            "selection_policy": "Figures prefer Step 05 nested-CV/global-best outputs and Step 07 validation-selected MIL outputs when available.",
        },
    }
    summary_path = METADATA_DIR / "step13_publication_summary.json"
    save_json(summary, summary_path)
    artifacts["step13_publication_summary_json"] = str(summary_path)

    if cfg.save_zip:
        zip_path = REPORTS_DIR / "step13_publication_figures.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(OUTPUT_DIR.glob("*")):
                if p.is_file():
                    zf.write(p, arcname=p.name)
            zf.write(summary_path, arcname=f"metadata/{summary_path.name}")
        artifacts["step13_publication_figures_zip"] = str(zip_path)

    state.mark_step_done(STEP_NAME, artifacts)
    state.add_note("Step 13 integrated publication figure package completed.")
    _save_step_manifest(_manifest, _key_inputs, "v1")

    logger.info("=== STEP 13 FIGURE GENERATION SUMMARY ===")
    for r in results:
        status = "OK " if r.get("success") else "FAIL"
        dest = r.get("output_path", r.get("reason", ""))
        logger.info("  %s  %-42s  %s", status, r["figure"], dest)
    logger.info("All figures saved to: %s", OUTPUT_DIR)
    logger.info("Completed %s successfully.", STEP_NAME)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
