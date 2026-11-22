#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Step 11 — OpenBTAI External Proxy Validation
============================================

Purpose
-------
Validate the BCBM leakage-aware Step 05 tabular radiomics pipeline on an
independent OpenBTAI breast-brain-metastasis cohort using receptor-proxy labels
inferred from breast molecular subtype.

Scientific interpretation
-------------------------
This is NOT direct external receptor validation. It is external PROXY validation:
OpenBTAI labels are derived from molecular subtype, e.g. Luminal A/B -> ER proxy
positive, HER2-enriched -> HER2 proxy positive, Triple Negative -> ER/PR/HER2
proxy negative.

Default inputs
--------------
BCBM internal data:
    bcbm_project/data/processed/analysis_ready_step04_patient_level.csv
    bcbm_project/metadata/step04_feature_dictionary.json
    bcbm_project/reports/tables/step05master_model_results.csv
    bcbm_project/reports/tables/step05master_global_best_models_by_target.csv

OpenBTAI external file:
    bcbm_project/data/external_ready/openbtai_breast_only_patient_level_proxy_ready.csv

Outputs
-------
    bcbm_project/reports/tables/step11_openbtai_external_proxy_results.csv
    bcbm_project/reports/tables/step11_openbtai_external_proxy_predictions.csv
    bcbm_project/reports/tables/step11_openbtai_feature_mapping.csv
    bcbm_project/metadata/step11_openbtai_external_proxy_summary.json
    bcbm_project/models/step11_external_refit/{target}/refit_internal_model_for_external.joblib
    bcbm_project/reports/figures/step11_openbtai_*.png

Run
---
    python run_pipeline.py  (runs Step 10 then Step 11 automatically)

Optional standalone:
    python src/external/step_11_openbtai_external_proxy_validation.py
    python src/external/step_11_openbtai_external_proxy_validation.py --external-csv path/to/openbtai_breast_only_patient_level_proxy_ready.csv
    python src/external/step_11_openbtai_external_proxy_validation.py --force

Why this is a separate Step 11
------------------------------
Step 05 intentionally keeps the locked internal test set separate and saves
summary/result tables, but not reusable fitted model objects. Step 11 therefore
rebuilds the selected Step 05 model family on BCBM train+valid patients only and
applies it once to OpenBTAI external proxy labels.
"""

import argparse
import importlib.util
import json
import joblib
import logging
import os
import random
import re
import sys
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("./bcbm_project").resolve()
STEP_NAME = "step_11_openbtai_external_proxy_validation"

INPUT_BCBM_PATIENT = PROJECT_ROOT / "data" / "processed" / "analysis_ready_step04_patient_level.csv"
INPUT_FEATURE_DICT = PROJECT_ROOT / "metadata" / "step04_feature_dictionary.json"
INPUT_STEP05_RESULTS = PROJECT_ROOT / "reports" / "tables" / "step05master_model_results.csv"
INPUT_STEP05_GLOBAL_BEST = PROJECT_ROOT / "reports" / "tables" / "step05master_global_best_models_by_target.csv"

DEFAULT_EXTERNAL_CSV = PROJECT_ROOT / "data" / "external_ready" / "openbtai_breast_only_patient_level_proxy_ready.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"
METADATA_DIR = PROJECT_ROOT / "metadata"
LOGS_DIR = PROJECT_ROOT / "logs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
MODEL_REFIT_DIR = PROJECT_ROOT / "models" / "step11_external_refit"

LOCAL_STEP05 = Path(__file__).resolve().parent.parent / "modeling" / "step_05_tabular_modeling.py"

TARGET_TO_EXTERNAL_LABEL = {
    "target_er": "external_er_proxy",
    "target_pr": "external_pr_proxy",
    "target_her2": "external_her2_proxy",
}
TARGET_DISPLAY = {"target_er": "ER", "target_pr": "PR", "target_her2": "HER2"}

GLOBAL_SEED = 42


@dataclass
class Step11Config:
    project_root: str = str(PROJECT_ROOT)
    step_name: str = STEP_NAME
    external_csv: str = str(DEFAULT_EXTERNAL_CSV)
    # radiomics_only was retired when step 04 began writing three explicit blocks; step 05 no
    # longer produces a variant by that name, so this default named nothing.
    preferred_feature_set_variant: str = "radiomics_pure"
    target_columns: List[str] = field(default_factory=lambda: ["target_er", "target_pr", "target_her2"])
    train_splits: List[str] = field(default_factory=lambda: ["train", "valid"])
    min_external_total: int = 6
    min_external_class_count: int = 1
    n_bootstrap: int = 5000
    global_seed: int = GLOBAL_SEED
    use_step05_selected_model_family: bool = True
    external_threshold: float = 0.5
    save_plots: bool = True
    force: bool = False
    no_cache: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Generic helpers
# ─────────────────────────────────────────────────────────────────────────────
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logger(log_file: Path) -> logging.Logger:
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
    logger.info("=" * 80)
    logger.info("NEW SESSION STARTED at %s", datetime.utcnow().isoformat())
    logger.info("=" * 80)
    return logger


def import_step05_module():
    if not LOCAL_STEP05.exists():
        raise FileNotFoundError(
            f"Could not find step_05_tabular_modeling.py at: {LOCAL_STEP05}. "
            "Expected at src/modeling/step_05_tabular_modeling.py relative to the project root."
        )
    spec = importlib.util.spec_from_file_location("step05_for_step11", LOCAL_STEP05)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import Step 05 from {LOCAL_STEP05}")
    mod = importlib.util.module_from_spec(spec)
    # Required for dataclasses in dynamically imported modules on Python 3.10+
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# Feature matching between BCBM Step 04 and OpenBTAI patient table
# ─────────────────────────────────────────────────────────────────────────────
def normalize_feature_key(name: Any) -> str:
    """
    Map BCBM patient-level names and OpenBTAI ext_* names to a common key.

    Examples:
      BCBM:    patient_original_firstorder_energy_mean -> original_firstorder_energy_mean
      OpenBTAI: ext_original_firstorder_energy_mean    -> original_firstorder_energy_mean
    """
    s = str(name).strip().lower()
    s = s.replace(" ", "_").replace("-", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")

    prefixes = [
        "patient_",
        "case_",
        "ext_",
        "openbtai_",
    ]
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if s.startswith(p):
                s = s[len(p):]
                changed = True

    # Some files encode spaces/case differently. Normalize common PyRadiomics tokens.
    token_map = {
        "first_order": "firstorder",
        "gray_level": "graylevel",
        "grey_level": "graylevel",
    }
    for a, b in token_map.items():
        s = s.replace(a, b)

    return s


def build_external_feature_map(bcbm_feature_cols: Sequence[str], external_cols: Sequence[str]) -> pd.DataFrame:
    ext_key_to_col: Dict[str, str] = {}
    for c in external_cols:
        key = normalize_feature_key(c)
        if key not in ext_key_to_col:
            ext_key_to_col[key] = c

    rows = []
    for bcol in bcbm_feature_cols:
        key = normalize_feature_key(bcol)
        ecol = ext_key_to_col.get(key)
        rows.append({
            "bcbm_feature": bcol,
            "match_key": key,
            "openbtai_feature": ecol,
            "matched": bool(ecol is not None),
        })
    return pd.DataFrame(rows)


def align_external_to_bcbm_features(
    bcbm_df: pd.DataFrame,
    external_df: pd.DataFrame,
    bcbm_feature_cols: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    mapping = build_external_feature_map(bcbm_feature_cols, external_df.columns)
    matched_bcbm = mapping.loc[mapping["matched"], "bcbm_feature"].tolist()

    # Build external matrix in one pass to avoid pandas fragmentation warnings.
    matched_map = mapping[mapping["matched"]].copy()
    if matched_map.empty:
        X_ext = pd.DataFrame(index=external_df.index)
    else:
        ext_data = {
            row["bcbm_feature"]: pd.to_numeric(external_df[row["openbtai_feature"]], errors="coerce")
            for _, row in matched_map.iterrows()
        }
        X_ext = pd.DataFrame(ext_data, index=external_df.index)

    # Train data must use exactly the matched BCBM columns in the same order.
    # Convert the whole block at once to avoid repeated column assignment.
    X_bcbm = bcbm_df[matched_bcbm].apply(pd.to_numeric, errors="coerce").copy()
    X_bcbm = X_bcbm.replace([np.inf, -np.inf], np.nan)
    X_ext = X_ext.reindex(columns=matched_bcbm).replace([np.inf, -np.inf], np.nan).copy()
    return X_bcbm, X_ext, matched_bcbm


# ─────────────────────────────────────────────────────────────────────────────
# Metrics / plots
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(probs, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "AUROC": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "AUPRC": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "Balanced_Acc": float(balanced_accuracy_score(y, pred)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Recall_Sensitivity": float(recall_score(y, pred, zero_division=0)),
        "Specificity": float(tn / max(tn + fp, 1)),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def bootstrap_metric_ci(
    y_true: np.ndarray,
    probs: np.ndarray,
    metric: str,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> Tuple[float, float, int]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(probs, dtype=float)
    rng = np.random.default_rng(seed)
    vals: List[float] = []
    n = len(y)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            if metric == "auroc":
                vals.append(float(roc_auc_score(yb, pb)))
            elif metric == "auprc":
                vals.append(float(average_precision_score(yb, pb)))
        except Exception:
            continue
    if len(vals) < 20:
        return float("nan"), float("nan"), int(len(vals))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), int(len(vals))


def save_roc_plot(y_true: np.ndarray, probs: np.ndarray, path: Path, title: str) -> None:
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc = roc_auc_score(y_true, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUROC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_pr_plot(y_true: np.ndarray, probs: np.ndarray, path: Path, title: str) -> None:
    if len(np.unique(y_true)) < 2:
        return
    precision, recall, _ = precision_recall_curve(y_true, probs)
    ap = average_precision_score(y_true, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, lw=2, label=f"AUPRC = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_summary_bar(results: pd.DataFrame, path: Path) -> None:
    d = results.dropna(subset=["AUROC"]).copy()
    if d.empty:
        return
    labels = d["target_display"].astype(str).tolist()
    vals = d["AUROC"].astype(float).tolist()
    err_low = [max(0.0, v - l) if pd.notna(l) else 0.0 for v, l in zip(vals, d["AUROC_CI95_lower"])]
    err_high = [max(0.0, u - v) if pd.notna(u) else 0.0 for v, u in zip(vals, d["AUROC_CI95_upper"])]
    plt.figure(figsize=(7, 4.5))
    x = np.arange(len(labels))
    plt.bar(x, vals, yerr=[err_low, err_high], capsize=4)
    plt.axhline(0.5, linestyle="--", lw=1)
    plt.ylim(0, 1)
    plt.xticks(x, labels)
    plt.ylabel("External proxy AUROC")
    plt.title("OpenBTAI external proxy validation")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Model selection
# ─────────────────────────────────────────────────────────────────────────────
def choose_step05_model_row(
    step05_results: pd.DataFrame,
    step05_global_best: pd.DataFrame,
    target_col: str,
    preferred_variant: str,
) -> Dict[str, Any]:
    """
    Prefer the requested feature-set variant, because OpenBTAI has radiomics but
    not scanner covariates. If absent, fall back to global Step 05 best.
    """
    d = step05_results[step05_results["target"].eq(target_col)].copy()
    if not d.empty and "feature_set_variant" in d.columns:
        dv = d[d["feature_set_variant"].eq(preferred_variant)].copy()
        if not dv.empty:
            sort_cols = [c for c in ["nested_cv_auroc_mean", "AUROC"] if c in dv.columns]
            if sort_cols:
                dv = dv.sort_values(sort_cols, ascending=[False] * len(sort_cols))
            return dv.iloc[0].to_dict()
    gb = step05_global_best[step05_global_best["target"].eq(target_col)].copy()
    if not gb.empty:
        return gb.iloc[0].to_dict()
    if not d.empty:
        return d.iloc[0].to_dict()
    raise ValueError(f"No Step 05 model result found for {target_col}")


# ─────────────────────────────────────────────────────────────────────────────
# Main validation routine
# ─────────────────────────────────────────────────────────────────────────────
def run_external_proxy_validation(cfg: Step11Config, logger: logging.Logger) -> Dict[str, Any]:
    step05 = import_step05_module()

    # Required inputs.
    required = [INPUT_BCBM_PATIENT, INPUT_FEATURE_DICT, INPUT_STEP05_RESULTS, Path(cfg.external_csv)]
    missing = [str(p) for p in required if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s):\n- " + "\n- ".join(missing))

    bcbm = pd.read_csv(INPUT_BCBM_PATIENT)
    external = pd.read_csv(cfg.external_csv)
    feature_dict = load_json(INPUT_FEATURE_DICT)
    step05_results = pd.read_csv(INPUT_STEP05_RESULTS)
    step05_global = pd.read_csv(INPUT_STEP05_GLOBAL_BEST) if INPUT_STEP05_GLOBAL_BEST.exists() else pd.DataFrame()

    if "split" not in bcbm.columns:
        raise ValueError("BCBM patient-level table is missing split column.")

    logger.info("Loaded BCBM patient-level: %s", bcbm.shape)
    logger.info("Loaded OpenBTAI external patient-level: %s", external.shape)
    logger.info("Training splits for Step 11: %s", cfg.train_splits)

    # BCBM radiomic block, aligned to external patient-level features. Prefer the explicit
    # block written by step 04: numeric_feature_columns_no_scanner_covariates is the 737-column
    # set that still carries field strength under four other names, plus pixel spacing and slice
    # geometry, so it is not the radiomic arm the manuscript defines.
    rad_only_cols = [c for c in feature_dict.get("radiomic_block_columns", []) if c in bcbm.columns]
    if not rad_only_cols:
        rad_only_cols = [c for c in feature_dict.get("numeric_feature_columns_no_scanner_covariates", []) if c in bcbm.columns]
    if not rad_only_cols:
        raise ValueError("No BCBM radiomic features found in step04_feature_dictionary.json")

    train_mask = bcbm["split"].isin(cfg.train_splits)
    bcbm_train = bcbm.loc[train_mask].copy()
    if bcbm_train.empty:
        raise ValueError(f"No BCBM training rows found for splits={cfg.train_splits}")

    X_bcbm_all, X_external_all, matched_features = align_external_to_bcbm_features(
        bcbm_train,
        external,
        rad_only_cols,
    )

    feature_mapping = build_external_feature_map(rad_only_cols, external.columns)
    feature_mapping_path = TABLES_DIR / "step11_openbtai_feature_mapping.csv"
    feature_mapping.to_csv(feature_mapping_path, index=False, encoding="utf-8-sig")

    logger.info(
        "Feature matching: %d/%d BCBM radiomic-block patient features matched OpenBTAI external columns.",
        len(matched_features),
        len(rad_only_cols),
    )
    if len(matched_features) < 5:
        raise ValueError(
            f"Only {len(matched_features)} features matched between BCBM and OpenBTAI. "
            "Check that the OpenBTAI patient-level file was created by build_openbtai_external_proxy_dataset.py."
        )

    runtime_info = step05.detect_runtime_environment() if hasattr(step05, "detect_runtime_environment") else {"gpu_available": False}
    gpu = bool(runtime_info.get("gpu_available", False))
    step05_cfg = step05.Step05Config()

    result_rows: List[Dict[str, Any]] = []
    prediction_frames: List[pd.DataFrame] = []
    skipped: List[Dict[str, Any]] = []

    id_cols = [c for c in ["patient_id", "patient_id_str", "primary_tumor_name", "breast_subtype_name"] if c in external.columns]

    for target_col in cfg.target_columns:
        ext_label = TARGET_TO_EXTERNAL_LABEL.get(target_col)
        if ext_label is None or ext_label not in external.columns:
            skipped.append({"target": target_col, "reason": f"external label missing: {ext_label}"})
            logger.warning("Skipping %s: external label column missing: %s", target_col, ext_label)
            continue

        model_row = choose_step05_model_row(step05_results, step05_global, target_col, cfg.preferred_feature_set_variant)
        model_name = str(model_row.get("model", "elastic_net"))
        variant_used = str(model_row.get("feature_set_variant", cfg.preferred_feature_set_variant))

        data_train = bcbm_train[bcbm_train[target_col].notna()].copy()
        if data_train.empty or data_train[target_col].nunique(dropna=True) < 2:
            skipped.append({"target": target_col, "reason": "BCBM train labels unavailable or one-class"})
            continue

        y_train = data_train[target_col].astype(int)
        X_train = X_bcbm_all.loc[data_train.index, matched_features].copy()

        # External rows with proxy label.
        y_ext_series = pd.to_numeric(external[ext_label], errors="coerce")
        ext_mask = y_ext_series.notna()
        y_ext = y_ext_series.loc[ext_mask].astype(int).to_numpy()
        X_ext = X_external_all.loc[ext_mask, matched_features].copy()
        external_subset = external.loc[ext_mask].copy()

        n_pos = int(np.sum(y_ext == 1))
        n_neg = int(np.sum(y_ext == 0))
        if len(y_ext) < cfg.min_external_total or min(n_pos, n_neg) < cfg.min_external_class_count:
            reason = f"insufficient external labels: n={len(y_ext)}, pos={n_pos}, neg={n_neg}"
            skipped.append({"target": target_col, "reason": reason})
            logger.warning("Skipping %s: %s", target_col, reason)
            continue

        # Build the same Step 05 model family and fit on BCBM train+valid only.
        candidates = step05.build_candidate_models(step05_cfg, y_train, gpu)
        if model_name not in candidates:
            logger.warning("Step 05 model '%s' unavailable in this environment. Falling back to elastic_net.", model_name)
            model_name = "elastic_net"
        model = step05.build_candidate_models(step05_cfg, y_train, gpu)[model_name]

        logger.info(
            "Fitting target=%s using Step05 model=%s | BCBM train n=%d | External n=%d pos=%d neg=%d",
            target_col, model_name, len(y_train), len(y_ext), n_pos, n_neg,
        )
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_ext)[:, 1]
        preds = (probs >= cfg.external_threshold).astype(int)

        # Save refit model for reproducibility/audit.
        model_dir = MODEL_REFIT_DIR / target_col
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_dir / "refit_internal_model_for_external.joblib")

        metrics = compute_metrics(y_ext, probs, threshold=cfg.external_threshold)
        auroc_lo, auroc_hi, auroc_n = bootstrap_metric_ci(y_ext, probs, "auroc", cfg.n_bootstrap, cfg.global_seed)
        auprc_lo, auprc_hi, auprc_n = bootstrap_metric_ci(y_ext, probs, "auprc", cfg.n_bootstrap, cfg.global_seed + 11)

        row = {
            "target": target_col,
            "target_display": TARGET_DISPLAY.get(target_col, target_col),
            "external_label_column": ext_label,
            "scenario": "openbtai_external_proxy_validation",
            "feature_set_variant_requested": cfg.preferred_feature_set_variant,
            "feature_set_variant_reference_from_step05": variant_used,
            "model": model_name,
            "model_display": getattr(step05, "MODEL_DISPLAY_NAMES", {}).get(model_name, model_name),
            "training_source": "BCBM train+valid",
            "external_source": "OpenBTAI breast-only subtype-derived proxy labels",
            "n_bcbm_train": int(len(y_train)),
            "n_external": int(len(y_ext)),
            "external_positive": n_pos,
            "external_negative": n_neg,
            "external_reliability_flag": (
                "very_limited_unstable_auc" if len(y_ext) < 10 or min(n_pos, n_neg) <= 1
                else "limited_interpret_with_ci" if len(y_ext) < 30
                else "acceptable"
            ),
            "n_bcbm_candidate_features": int(len(rad_only_cols)),
            "n_matched_features": int(len(matched_features)),
            "matched_feature_ratio": float(len(matched_features) / max(len(rad_only_cols), 1)),
            "external_threshold": float(cfg.external_threshold),
            "AUROC_CI95_lower": auroc_lo,
            "AUROC_CI95_upper": auroc_hi,
            "AUROC_bootstrap_n": auroc_n,
            "AUPRC_CI95_lower": auprc_lo,
            "AUPRC_CI95_upper": auprc_hi,
            "AUPRC_bootstrap_n": auprc_n,
            "scientific_interpretation": "external proxy validation; labels are subtype-derived, not direct receptor assays",
            **metrics,
        }
        result_rows.append(row)

        pred_df = external_subset[id_cols].copy() if id_cols else pd.DataFrame(index=external_subset.index)
        pred_df["target"] = target_col
        pred_df["target_display"] = TARGET_DISPLAY.get(target_col, target_col)
        pred_df["external_label_column"] = ext_label
        pred_df["y_true"] = y_ext
        pred_df["y_true_proxy"] = y_ext
        pred_df["probability"] = probs.astype(float)
        pred_df["prediction"] = preds.astype(int)
        pred_df["model"] = model_name
        pred_df["feature_set_variant"] = cfg.preferred_feature_set_variant
        pred_df["combat_mode"] = "none"
        prediction_frames.append(pred_df)

        if cfg.save_plots:
            st = re.sub(r"[^a-z0-9_]+", "_", target_col.lower())
            save_roc_plot(
                y_ext, probs,
                FIGURES_DIR / f"step11_openbtai_{st}_roc.png",
                f"OpenBTAI proxy validation — {TARGET_DISPLAY.get(target_col, target_col)} ROC",
            )
            save_pr_plot(
                y_ext, probs,
                FIGURES_DIR / f"step11_openbtai_{st}_pr.png",
                f"OpenBTAI proxy validation — {TARGET_DISPLAY.get(target_col, target_col)} PR",
            )

    results_df = pd.DataFrame(result_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    skipped_df = pd.DataFrame(skipped)

    results_path = TABLES_DIR / "step11_openbtai_external_proxy_results.csv"
    predictions_path = TABLES_DIR / "step11_openbtai_external_proxy_predictions.csv"
    skipped_path = TABLES_DIR / "step11_openbtai_external_proxy_skipped_targets.csv"

    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    predictions_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    skipped_df.to_csv(skipped_path, index=False, encoding="utf-8-sig")

    if cfg.save_plots and not results_df.empty:
        save_summary_bar(results_df, FIGURES_DIR / "step11_openbtai_external_proxy_auroc_summary.png")

    summary = {
        "config": asdict(cfg),
        "runtime": runtime_info,
        "created_at": datetime.utcnow().isoformat(),
        "inputs": {
            "bcbm_patient_level": str(INPUT_BCBM_PATIENT),
            "feature_dictionary": str(INPUT_FEATURE_DICT),
            "step05_results": str(INPUT_STEP05_RESULTS),
            "step05_global_best": str(INPUT_STEP05_GLOBAL_BEST),
            "external_csv": str(cfg.external_csv),
        },
        "feature_matching": {
            "bcbm_radiomics_only_features": int(len(rad_only_cols)),
            "matched_features": int(len(matched_features)),
            "mapping_csv": str(feature_mapping_path),
        },
        "results_csv": str(results_path),
        "predictions_csv": str(predictions_path),
        "skipped_targets_csv": str(skipped_path),
        "n_results": int(len(results_df)),
        "skipped_targets": skipped,
        "method_notes": {
            "training": "Models are rebuilt on BCBM train+valid patients only using the Step 05 model family.",
            "external_testing": "OpenBTAI is used once as an independent external proxy cohort.",
            "features": "Only radiomics-only matched patient-level features are used; scanner covariates are intentionally excluded.",
            "labels": "External labels are receptor proxies derived from breast molecular subtype.",
            "not_direct_validation": "Do not describe this as direct ER/PR/HER2 external validation.",
        },
        "results": results_df.to_dict(orient="records") if not results_df.empty else [],
    }
    return summary


class StateManager:
    def __init__(self, state_path: Path):
        self.state_path = state_path
        self.state = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"steps_completed": [], "artifacts": {}, "notes": []}

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state["last_updated"] = datetime.utcnow().isoformat()
        self.state_path.write_text(json.dumps(self.state, indent=2, ensure_ascii=False), encoding="utf-8")

    def mark_step_done(self, step_name: str, artifacts: Optional[Dict[str, str]] = None) -> None:
        if step_name not in self.state.setdefault("steps_completed", []):
            self.state["steps_completed"].append(step_name)
        if artifacts:
            self.state.setdefault("artifacts", {}).update(artifacts)
        self.save()

    def add_note(self, note: str) -> None:
        self.state.setdefault("notes", []).append({"time": datetime.utcnow().isoformat(), "note": note})
        self.save()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 11 OpenBTAI external proxy validation")
    p.add_argument("--external-csv", type=str, default=str(DEFAULT_EXTERNAL_CSV), help="OpenBTAI breast-only patient-level proxy CSV")
    p.add_argument("--preferred-feature-set", type=str, default="radiomics_pure", help="Step 05 feature set to mimic; default radiomics_pure (the 642-column radiomic block)")
    p.add_argument("--threshold", type=float, default=0.5, help="External classification threshold for threshold-dependent metrics")
    p.add_argument("--bootstrap", type=int, default=5000, help="Bootstrap iterations for AUROC/AUPRC CI")
    p.add_argument("--no-plots", action="store_true", help="Disable ROC/PR/summary plots")
    p.add_argument("--force",    action="store_true", help="Re-run even if outputs already exist")
    p.add_argument("--no-cache", action="store_true", help="Disable cache check (same as --force)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Step11Config(
        external_csv=str(Path(args.external_csv)),
        preferred_feature_set_variant=str(args.preferred_feature_set),
        external_threshold=float(args.threshold),
        n_bootstrap=int(args.bootstrap),
        save_plots=not bool(args.no_plots),
        force=bool(args.force),
        no_cache=bool(args.no_cache),
    )
    set_seed(cfg.global_seed)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    for p in [REPORTS_DIR, TABLES_DIR, FIGURES_DIR, METADATA_DIR, LOGS_DIR, CHECKPOINTS_DIR, MODEL_REFIT_DIR]:
        ensure_dir(p)

    logger = setup_logger(LOGS_DIR / f"{STEP_NAME}.log")
    logger.info("Starting %s", STEP_NAME)
    logger.info("External CSV: %s", cfg.external_csv)

    # Cache skip: if outputs already exist and --force not requested, skip.
    summary_path = METADATA_DIR / "step11_openbtai_external_proxy_summary.json"
    results_path = TABLES_DIR / "step11_openbtai_external_proxy_results.csv"
    if not cfg.force and not cfg.no_cache and summary_path.exists() and results_path.exists():
        logger.info(
            "Step 11 outputs already exist. Skipping (use --force to re-run)."
        )
        return

    summary = run_external_proxy_validation(cfg, logger)
    save_json(summary, summary_path)

    state = StateManager(CHECKPOINTS_DIR / "pipeline_state.json")
    state.mark_step_done(STEP_NAME, artifacts={
        "step11_summary_json": str(summary_path),
        "step11_results_csv": summary.get("results_csv", ""),
        "step11_predictions_csv": summary.get("predictions_csv", ""),
        "step11_feature_mapping_csv": summary.get("feature_matching", {}).get("mapping_csv", ""),
    })
    state.add_note("Step 11 OpenBTAI external proxy validation completed.")

    logger.info("Completed %s successfully.", STEP_NAME)
    logger.info("Summary: %s", summary_path)
    logger.info("=" * 80)

    # Console-friendly final table.
    results = pd.DataFrame(summary.get("results", []))
    if not results.empty:
        cols = [
            "target_display", "model", "n_external", "external_positive", "external_negative",
            "n_matched_features", "external_reliability_flag", "AUROC", "AUROC_CI95_lower", "AUROC_CI95_upper", "AUPRC",
        ]
        cols = [c for c in cols if c in results.columns]
        print("\nSTEP 11 OpenBTAI external proxy validation results")
        print(results[cols].to_string(index=False))
    else:
        print("\nSTEP 11 completed, but no valid target produced metrics. Check skipped target CSV.")


if __name__ == "__main__":
    main()
