"""
Step 08A — Multiple Instance Learning (MIL) for Receptor Status Prediction
===========================================================================
Dataset reality (from BCBM_Full_Data.xlsx):
  - 165 patients, 2825 lesion rows
  - 117 patients have >= 2 lesions  → MIL is the scientifically correct approach
  - Labels are at PATIENT level (ER/PR/HER2), not lesion level
  - Each patient = a "bag" of lesion radiomics instances

Architecture:
  - Attention-MIL  (Ilse et al. 2018)  — learns which lesions drive the prediction
  - Gated Attention-MIL                — more expressive gating mechanism
  - Both trained on radiomics features from existing pipeline (no NIfTI required)

Why MIL here:
  - Correct for patient-level labels with multiple lesions per patient
  - Preserves lesion-level information without aggregation loss (unlike Step 04 mean/std)
  - Attention weights give per-lesion biological interpretability
  - Works on 107 radiomics features → no raw images needed

GPU: RTX 4050 (CUDA) is used automatically when available.

Inputs  (from Step 03 output):
  data/processed/analysis_ready_step03_lesion_only.csv
  metadata/step04_feature_dictionary.json
  metadata/step04_patient_splits.csv

Outputs:
  reports/tables/step07_mil_*.csv
  reports/figures/step07_mil_*.png
  metadata/step07_mil_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import warnings
import zipfile
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, confusion_matrix, matthews_corrcoef
)
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore", category=UserWarning)

# ── PyTorch ─────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    raise RuntimeError(
        "PyTorch is required for Step 08A.\n"
        "Install with: pip install torch --index-url https://download.pytorch.org/whl/cu121"
    )

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("./bcbm_project").resolve()
STEP_NAME    = "step_07_mil_radiomics"

INPUT_LESION_CSV   = PROJECT_ROOT / "data" / "processed" / "analysis_ready_step03_lesion_only.csv"
INPUT_FEATURE_DICT = PROJECT_ROOT / "metadata" / "step04_feature_dictionary.json"
INPUT_SPLITS_CSV   = PROJECT_ROOT / "metadata" / "step04_patient_splits.csv"

REPORTS_DIR    = PROJECT_ROOT / "reports"
FIGURES_DIR    = REPORTS_DIR  / "figures"
TABLES_DIR     = REPORTS_DIR  / "tables"
METADATA_DIR   = PROJECT_ROOT / "metadata"
LOGS_DIR       = PROJECT_ROOT / "logs"
CHECKPOINTS_DIR= PROJECT_ROOT / "checkpoints"

TARGET_COLUMNS      = ["target_er", "target_pr", "target_her2"]
LABEL_SOURCE_COLS   = {"target_er": "ER_bin", "target_pr": "PR_bin", "target_her2": "HER2_bin"}
TARGET_DISPLAY      = {"target_er": "ER", "target_pr": "PR", "target_her2": "HER2"}
GLOBAL_SEED         = 42
MAX_INSTANCES_PER_BAG = 30   # cap bags to avoid domination by outlier patients (max=949)


# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Step08AConfig:
    project_root:        str   = str(PROJECT_ROOT)
    step_name:           str   = STEP_NAME
    target_columns:      List[str] = field(default_factory=lambda: TARGET_COLUMNS.copy())
    global_seed:         int   = GLOBAL_SEED
    max_instances:       int   = MAX_INSTANCES_PER_BAG
    # MIL architecture
    hidden_dim:          int   = 128
    attention_dim:       int   = 64
    dropout:             float = 0.3
    # Training
    n_epochs:            int   = 150      # FIX: increased to allow proper convergence
    lr:                  float = 1e-3
    weight_decay:        float = 1e-4
    patience:            int   = 35       # FIX: increased from 25 — give HER2 more time
    batch_size:          int   = 16      # bags per batch
    min_valid_specificity: float = 0.20   # FIX: minimum specificity for threshold selection
    # Augmentation
    use_feature_noise:   bool  = True
    noise_std:           float = 0.02
    # Prediction threshold
    threshold_grid: List[float] = field(default_factory=lambda: [0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65])
    n_bootstrap:         int   = 1000
    save_plots:          bool  = True
    save_package:        bool  = True


# ════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def load_json(path: Path) -> Dict:
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


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger(STEP_NAME)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh  = logging.FileHandler(log_file, encoding="utf-8")
    sh  = logging.StreamHandler()
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    from datetime import datetime
    logger.info("=" * 70)
    logger.info("NEW SESSION — %s", datetime.utcnow().isoformat())
    logger.info("=" * 70)
    return logger


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    return dev


# ════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ════════════════════════════════════════════════════════════════════════════

def load_feature_cols(lesion_df: pd.DataFrame, feature_dict_path: Path) -> List[str]:
    """Load radiomics feature columns (no scanner covariates, no metadata)."""
    if feature_dict_path.exists():
        fd = load_json(feature_dict_path)
        # Prefer radiomics-only (no scanner confounders) for the MIL input
        cols = [c for c in fd.get("numeric_feature_columns_no_scanner_covariates", [])
                if c in lesion_df.columns]
        if cols:
            return cols
    # Fallback: detect radiomics columns by prefix
    prefixes = ("original_", "wavelet_", "log_sigma_", "shape_",
                "firstorder_", "glcm_", "glrlm_", "glszm_", "gldm_", "ngtdm_")
    return [c for c in lesion_df.columns
            if pd.api.types.is_numeric_dtype(lesion_df[c])
            and any(c.startswith(p) for p in prefixes)]


def normalize_label_pm(x) -> Optional[int]:
    """Convert '+'/'-'/0/1 to int binary label."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s in ("+", "1", "1.0"):
        return 1
    if s in ("-", "0", "0.0"):
        return 0
    return None


def build_bags(
    lesion_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    splits_df: pd.DataFrame,
    cfg: Step08AConfig,
    scaler: Optional[RobustScaler] = None,
    imputer: Optional[SimpleImputer] = None,
    fit_preprocessor: bool = False,
) -> Tuple[List[Dict], RobustScaler, SimpleImputer]:
    """
    Build patient-level bags. Each bag = {features: (N_lesions, D), label: int, patient_id, split}.
    Applies instance-count capping to prevent single patient domination.
    """
    label_src = LABEL_SOURCE_COLS.get(target_col, target_col.replace("target_", "") + "_bin")

    # Merge splits
    df = lesion_df.copy()
    df = df.merge(splits_df[["patient_base", "split"]], on="patient_base", how="left")
    # Keep only rows with explicit split assignments. Split integrity is also checked in main().
    df = df[df["split"].isin(["train", "valid", "test"])].copy()
    df = df[df[label_src].notna()].copy()
    df["_label"] = df[label_src].apply(normalize_label_pm)
    df = df[df["_label"].notna()].copy()

    # Fit preprocessor on train bags only
    X_all = df[feature_cols].copy()
    X_all = X_all.replace([np.inf, -np.inf], np.nan)
    X_all = X_all.apply(pd.to_numeric, errors="coerce")

    if fit_preprocessor:
        train_mask = df["split"] == "train"
        if train_mask.sum() == 0:
            raise ValueError(f"No training rows available for {target_col} after filtering.")
        imputer = SimpleImputer(strategy="median")
        imputer.fit(X_all.loc[train_mask])
        X_imp = imputer.transform(X_all)
        scaler = RobustScaler()
        scaler.fit(X_imp[train_mask.values])
    else:
        if imputer is None or scaler is None:
            raise ValueError("imputer and scaler must be provided when fit_preprocessor=False")
        X_imp = imputer.transform(X_all)

    X_scaled = scaler.transform(X_imp)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    df_feat  = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)

    bags = []
    for patient_id, grp in df.groupby("patient_base"):
        label_vals = grp["_label"].unique()
        if len(label_vals) > 1:
            continue  # skip label-conflicted patients
        label = int(label_vals[0])
        split_vals = grp["split"].dropna().unique() if "split" in grp.columns else ["train"]
        if len(split_vals) != 1:
            continue  # skip patients with inconsistent split assignments
        split = str(split_vals[0])

        feats = df_feat.loc[grp.index].values.astype(np.float32)
        # Cap instances to avoid 949-instance outlier bags
        if len(feats) > cfg.max_instances:
            idx = np.random.choice(len(feats), cfg.max_instances, replace=False)
            feats = feats[idx]

        bags.append({
            "features":   feats,
            "label":      label,
            "patient_id": patient_id,
            "split":      split,
            "n_instances": len(feats),
        })

    return bags, scaler, imputer


# ════════════════════════════════════════════════════════════════════════════
# PYTORCH DATASET
# ════════════════════════════════════════════════════════════════════════════

class BagDataset(Dataset):
    """Each item = one patient bag (variable-length set of lesion feature vectors)."""

    def __init__(self, bags: List[Dict], augment: bool = False, noise_std: float = 0.02):
        self.bags     = bags
        self.augment  = augment
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        bag   = self.bags[idx]
        feats = torch.from_numpy(bag["features"].copy())   # (N, D)
        label = torch.tensor(bag["label"], dtype=torch.float32)
        if self.augment and self.noise_std > 0:
            feats = feats + torch.randn_like(feats) * self.noise_std
        return feats, label


def collate_bags(batch):
    """Pad variable-length bags to same length within a batch."""
    feats_list, labels = zip(*batch)
    max_n = max(f.shape[0] for f in feats_list)
    d     = feats_list[0].shape[1]
    padded = torch.zeros(len(feats_list), max_n, d)
    mask   = torch.zeros(len(feats_list), max_n, dtype=torch.bool)   # True = real
    for i, f in enumerate(feats_list):
        n = f.shape[0]
        padded[i, :n] = f
        mask[i, :n]   = True
    labels = torch.stack(labels)
    return padded, mask, labels


# ════════════════════════════════════════════════════════════════════════════
# MIL MODELS
# ════════════════════════════════════════════════════════════════════════════

class AttentionMIL(nn.Module):
    """
    Attention-Based MIL (Ilse et al. 2018).
    Learns per-instance attention weights → weighted sum → classifier.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 attention_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Attention network
        self.attention_V  = nn.Linear(hidden_dim, attention_dim)
        self.attention_U  = nn.Linear(hidden_dim, attention_dim)
        self.attention_w  = nn.Linear(attention_dim, 1)
        # Classifier
        self.classifier   = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        x    : (B, N, D)
        mask : (B, N)  True = real instance, False = padding
        """
        B, N, D = x.shape
        h = self.feature_extractor(x.view(B * N, D)).view(B, N, -1)  # (B, N, H)

        # Attention weights
        A_V = torch.tanh(self.attention_V(h))                         # (B, N, att_dim)
        A_U = torch.sigmoid(self.attention_U(h))                      # (B, N, att_dim)
        A   = self.attention_w(A_V * A_U).squeeze(-1)                 # (B, N)

        # Mask padding
        A = A.masked_fill(~mask, float("-inf"))
        A = F.softmax(A, dim=1)                                       # (B, N)
        A = torch.nan_to_num(A, nan=0.0)

        # Aggregate
        z = (A.unsqueeze(-1) * h).sum(dim=1)                         # (B, H)
        logit = self.classifier(z).squeeze(-1)                        # (B,)
        return logit, A                                               # return attention for viz


class GatedAttentionMIL(nn.Module):
    """
    Gated Attention-MIL — separate tanh & sigmoid branches (more expressive).
    Better for heterogeneous lesion populations.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 attention_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attn_tanh    = nn.Linear(hidden_dim, attention_dim)
        self.attn_sigmoid = nn.Linear(hidden_dim, attention_dim)
        self.attn_weight  = nn.Linear(attention_dim, 1)
        self.classifier   = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        B, N, D = x.shape
        h = self.encoder(x.view(B * N, D)).view(B, N, -1)
        A = self.attn_weight(torch.tanh(self.attn_tanh(h)) * torch.sigmoid(self.attn_sigmoid(h))).squeeze(-1)
        A = A.masked_fill(~mask, float("-inf"))
        A = F.softmax(A, dim=1)
        A = torch.nan_to_num(A, nan=0.0)
        z = (A.unsqueeze(-1) * h).sum(dim=1)
        logit = self.classifier(z).squeeze(-1)
        return logit, A


# ════════════════════════════════════════════════════════════════════════════
# TRAINING ENGINE
# ════════════════════════════════════════════════════════════════════════════

def compute_pos_weight(bags: List[Dict], device: torch.device) -> torch.Tensor:
    labels = [b["label"] for b in bags]
    n_pos  = sum(labels)
    n_neg  = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return torch.tensor(1.0, device=device)
    return torch.tensor(n_neg / n_pos, dtype=torch.float32, device=device)


def train_epoch(model, loader, optimizer, criterion, device, augment_noise=0.0):
    model.train()
    total_loss = 0.0
    for feats, mask, labels in loader:
        feats, mask, labels = feats.to(device), mask.to(device), labels.to(device)
        if augment_noise > 0:
            feats = feats + torch.randn_like(feats) * augment_noise
        optimizer.zero_grad()
        logits, _ = model(feats, mask)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_labels, all_pids = [], [], []
    for feats, mask, labels in loader:
        feats, mask = feats.to(device), mask.to(device)
        logits, _ = model(feats, mask)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().tolist())
    return np.array(all_probs), np.array(all_labels, dtype=int)


def choose_threshold(y_true, probs, grid, min_specificity: float = 0.20):
    """
    FIX: Choose threshold on validation set using Balanced Accuracy,
    with a minimum specificity constraint to prevent degenerate
    "predict everything positive" solutions.

    min_specificity=0.20 means the model must correctly identify
    at least 20% of negatives — prevents Sensitivity=1, Specificity=0.
    Falls back to best BA without constraint if no threshold meets it.
    """
    best_t, best_ba   = 0.5, -1.0
    best_t_fb, best_fb = 0.5, -1.0   # fallback without constraint

    for t in grid:
        preds = (probs >= t).astype(int)
        ba    = balanced_accuracy_score(y_true, preds)

        # Check specificity constraint
        tn = int(((y_true == 0) & (preds == 0)).sum())
        fp = int(((y_true == 0) & (preds == 1)).sum())
        spec = tn / max(tn + fp, 1)

        # Track unconstrained best (fallback)
        if ba > best_fb:
            best_fb, best_t_fb = ba, t

        # Track constrained best
        if ba > best_ba and spec >= min_specificity:
            best_ba, best_t = ba, t

    # If no threshold met the constraint, use unconstrained best
    if best_ba < 0:
        return best_t_fb
    return best_t


def compute_metrics(y_true, probs, preds):
    if len(np.unique(y_true)) < 2:
        return {"AUROC": float("nan"), "AUPRC": float("nan"), "F1": float("nan"),
                "Balanced_Acc": float("nan")}
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    sens = float(tp / max(tp + fn, 1))
    spec = float(tn / max(tn + fp, 1))

    # FIX: Flag degenerate predictions for clinical awareness
    is_degenerate = (spec < 0.05) or (sens < 0.05)

    return {
        "AUROC":            float(roc_auc_score(y_true, probs)),
        "AUPRC":            float(average_precision_score(y_true, probs)),
        "F1":               float(f1_score(y_true, preds, zero_division=0)),
        "Balanced_Acc":     float(balanced_accuracy_score(y_true, preds)),
        "Sensitivity":      sens,
        "Specificity":      spec,
        "PPV_Precision":    float(tp / max(tp + fp, 1)),
        "NPV":              float(tn / max(tn + fn, 1)),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "degenerate_prediction": bool(is_degenerate),
        "MCC": float(matthews_corrcoef(y_true, preds)),
    }


def bootstrap_auroc(y_true, probs, n=1000, seed=42) -> Tuple[float, float]:
    rng  = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        yb, pb = y_true[idx], probs[idx]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(roc_auc_score(yb, pb))
    if len(aucs) < 100:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def train_mil(
    model_class,
    train_bags, valid_bags, test_bags,
    feature_dim: int,
    cfg: Step08AConfig,
    device: torch.device,
    logger: logging.Logger,
    model_name: str,
) -> Dict[str, Any]:
    """Full train → early-stop on valid AUROC → final locked test evaluation."""

    model = model_class(
        input_dim=feature_dim,
        hidden_dim=cfg.hidden_dim,
        attention_dim=cfg.attention_dim,
        dropout=cfg.dropout,
    ).to(device)

    pos_weight = compute_pos_weight(train_bags, device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_epochs)

    train_ds = BagDataset(train_bags, augment=cfg.use_feature_noise, noise_std=cfg.noise_std)
    valid_ds = BagDataset(valid_bags)
    test_ds  = BagDataset(test_bags)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=collate_bags, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False,
                              collate_fn=collate_bags)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False,
                              collate_fn=collate_bags)

    best_composite_score  = -1.0   # composite = 0.7*AUROC + 0.3*BA - spec_penalty
    best_valid_auroc_real = -1.0   # real AUROC at best checkpoint
    # Initialize from the model's initial weights (never None) so that, if no epoch
    # ever improves the composite score, load_state_dict() restores valid weights
    # instead of crashing. Matches the project rule and steps 08/09.
    best_state            = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    patience_count        = 0
    history               = []

    for epoch in range(1, cfg.n_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device,
                                 augment_noise=cfg.noise_std if cfg.use_feature_noise else 0.0)
        scheduler.step()

        v_probs, v_labels = evaluate(model, valid_loader, device)
        v_auroc = roc_auc_score(v_labels, v_probs) if len(np.unique(v_labels)) > 1 else 0.5

        # FIX 3: Composite early-stopping score — prevents stopping when
        # AUROC is high but threshold produces degenerate predictions
        v_preds_05   = (v_probs >= 0.5).astype(int)
        v_ba_05      = balanced_accuracy_score(v_labels, v_preds_05)
        v_tn_05      = int(((v_labels == 0) & (v_preds_05 == 0)).sum())
        v_fp_05      = int(((v_labels == 0) & (v_preds_05 == 1)).sum())
        v_spec_05    = v_tn_05 / max(v_tn_05 + v_fp_05, 1)
        # Penalize when specificity at 0.5 is degenerate (< 0.1)
        spec_penalty = 0.1 if v_spec_05 < 0.1 else 0.0
        composite    = 0.7 * v_auroc + 0.3 * v_ba_05 - spec_penalty

        history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                        "valid_auroc":    round(v_auroc, 4),
                        "valid_composite": round(composite, 4)})

        if composite > best_composite_score:
            best_composite_score  = composite
            best_valid_auroc_real = v_auroc   # real AUROC at this checkpoint
            best_state            = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count        = 0
        else:
            patience_count += 1

        if epoch % 20 == 0:
            logger.info("  [%s] Epoch %3d | loss=%.4f | valid_AUROC=%.4f | composite=%.4f (best=%.4f)",
                        model_name, epoch, train_loss, v_auroc, composite, best_composite_score)

        if patience_count >= cfg.patience:
            logger.info("  [%s] Early stop at epoch %d (patience=%d)", model_name, epoch, cfg.patience)
            break

    # Restore best model. best_state is seeded from the initial weights before the
    # loop, so it is always a valid state_dict even if no epoch improved the score.
    model.load_state_dict(best_state)

    # Threshold on validation set ONLY
    v_probs, v_labels = evaluate(model, valid_loader, device)
    best_threshold = choose_threshold(v_labels, v_probs, cfg.threshold_grid,
                                       min_specificity=cfg.min_valid_specificity)
    valid_metrics  = compute_metrics(v_labels, v_probs, (v_probs >= best_threshold).astype(int))

    # ── LOCKED TEST EVALUATION ──────────────────────────────────────────
    t_probs, t_labels = evaluate(model, test_loader, device)
    t_preds   = (t_probs >= best_threshold).astype(int)
    test_metrics = compute_metrics(t_labels, t_probs, t_preds)
    ci_lo, ci_hi = bootstrap_auroc(t_labels, t_probs, n=cfg.n_bootstrap)
    test_metrics["AUROC_CI95_lower"] = ci_lo
    test_metrics["AUROC_CI95_upper"] = ci_hi
    test_metrics["generalization_gap"] = float(valid_metrics["AUROC"] - test_metrics["AUROC"])

    return {
        "model":                model,
        "model_name":           model_name,
        "history":              history,
        "best_composite_score": best_composite_score,
        "best_valid_auroc":     best_valid_auroc_real,  # real AUROC at best checkpoint (not composite)
        "best_threshold":  best_threshold,
        "valid_metrics":   valid_metrics,
        "test_metrics":    test_metrics,
        "test_probs":      t_probs,
        "test_labels":     t_labels,
        "valid_probs":     v_probs,
        "valid_labels":    v_labels,
    }


# ════════════════════════════════════════════════════════════════════════════
# ATTENTION ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def get_attention_weights(model, bags: List[Dict], device: torch.device) -> pd.DataFrame:
    """Extract per-lesion attention weights for interpretability."""
    model.eval()
    records = []
    for bag in bags:
        feats = torch.from_numpy(bag["features"]).unsqueeze(0).to(device)
        mask  = torch.ones(1, feats.shape[1], dtype=torch.bool, device=device)
        _, attn = model(feats, mask)
        weights = attn[0].cpu().numpy()  # (N,)
        for i, w in enumerate(weights):
            records.append({
                "patient_id":    bag["patient_id"],
                "label":         bag["label"],
                "split":         bag["split"],
                "instance_rank": i + 1,
                "attention_weight": float(w),
            })
    return pd.DataFrame(records)


# ════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ════════════════════════════════════════════════════════════════════════════

def plot_training_curves(history: List[Dict], path: Path, title: str) -> None:
    """3-panel training curves: loss, real AUROC, composite early-stopping score.
    Matches the same layout used in Steps 08B and 08C."""
    epochs = [h["epoch"] for h in history]
    losses = [h["train_loss"] for h in history]
    aurocs = [h["valid_auroc"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Panel 1: Training Loss
    axes[0].plot(epochs, losses, color="#2E75B6", lw=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss"); axes[0].grid(alpha=0.3)

    # Panel 2: Real AUROC (the clinically meaningful metric)
    axes[1].plot(epochs, aurocs, color="#375623", lw=2, label="Real AUROC")
    axes[1].axhline(0.5, linestyle="--", color="gray", alpha=0.5, label="Random")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("AUROC")
    axes[1].set_title("Validation AUROC (real)"); axes[1].set_ylim(0, 1)
    axes[1].legend(); axes[1].grid(alpha=0.3)

    # Panel 3: Composite early-stopping score (0.7×AUROC + 0.3×BA − spec_penalty)
    composites = [h.get("valid_composite", None) for h in history]
    if any(v is not None for v in composites):
        axes[2].plot(epochs, composites, color="#7B2C9E", lw=2, label="Composite")
        axes[2].axhline(0.5, linestyle="--", color="gray", alpha=0.5)
        axes[2].set_xlabel("Epoch")
        axes[2].set_title("Composite Score (early stopping)"); axes[2].set_ylim(0, 1)
        axes[2].legend(); axes[2].grid(alpha=0.3)
        axes[2].annotate("= 0.7×AUROC + 0.3×BA − spec_penalty",
                         xy=(0.02, 0.04), xycoords="axes fraction", fontsize=7, color="#7B2C9E")
    else:
        axes[2].axis("off")

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()


def plot_roc_pr(y_true, probs, path: Path, title: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc_val = roc_auc_score(y_true, probs)
    ax1.plot(fpr, tpr, lw=2, color="#2E75B6", label=f"AUROC = {auc_val:.3f}")
    ax1.plot([0,1],[0,1], "--", color="gray", lw=1)
    ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR"); ax1.set_title("ROC Curve")
    ax1.legend(); ax1.grid(alpha=0.3)
    prec, rec, _ = precision_recall_curve(y_true, probs)
    ap = average_precision_score(y_true, probs)
    ax2.plot(rec, prec, lw=2, color="#C00000", label=f"AUPRC = {ap:.3f}")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision"); ax2.set_title("PR Curve")
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()


def plot_attention_distribution(attn_df: pd.DataFrame, path: Path, title: str) -> None:
    """
    FIX 8: Enhanced attention plot — separates positive vs negative patients
    to reveal whether the model assigns different attention patterns by label.
    This is a key interpretability figure for publication.
    """
    if attn_df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel 1: Attention weight distribution by label
    for lbl, color, name in [(1, "#C00000", "Positive"), (0, "#2E75B6", "Negative")]:
        sub = attn_df[attn_df["label"] == lbl]["attention_weight"]
        if not sub.empty:
            axes[0].hist(sub, bins=30, alpha=0.6, color=color, edgecolor="white", label=name)
    axes[0].set_xlabel("Attention Weight")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Attention Distribution by Label")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Panel 2: Max attention per patient (top 20), colored by label
    top_bags = (attn_df.groupby(["patient_id", "label"])["attention_weight"]
                .max().reset_index()
                .sort_values("attention_weight", ascending=False).head(20))
    colors_bar = ["#C00000" if l == 1 else "#2E75B6" for l in top_bags["label"]]
    axes[1].barh(range(len(top_bags)), top_bags["attention_weight"].values,
                 color=colors_bar)
    axes[1].set_yticks(range(len(top_bags)))
    axes[1].set_yticklabels([f"P{int(r.patient_id)}" for _, r in top_bags.iterrows()], fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Max Attention Weight")
    axes[1].set_title("Top-20 Patients (Red=Pos, Blue=Neg)")
    axes[1].grid(alpha=0.3)

    # Panel 3: Mean max-attention per label (box plot)
    pos_attn = attn_df[attn_df["label"] == 1].groupby("patient_id")["attention_weight"].max()
    neg_attn = attn_df[attn_df["label"] == 0].groupby("patient_id")["attention_weight"].max()
    data_box = [pos_attn.values, neg_attn.values]
    bp = axes[2].boxplot(data_box, tick_labels=["Positive", "Negative"],
                         patch_artist=True, notch=False)
    bp["boxes"][0].set_facecolor("#C00000"); bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor("#2E75B6"); bp["boxes"][1].set_alpha(0.6)
    axes[2].set_ylabel("Max Attention Weight per Patient")
    axes[2].set_title("Attention by Receptor Status")
    axes[2].grid(alpha=0.3)

    # Add significance annotation if scipy available
    try:
        from scipy.stats import mannwhitneyu
        _, pval = mannwhitneyu(pos_attn, neg_attn, alternative="two-sided")
        axes[2].set_xlabel(f"Mann-Whitney p={pval:.3f}")
    except Exception:
        pass

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()


def plot_summary_comparison(all_results: List[Dict], path: Path) -> None:
    """Bar chart comparing all models across all targets."""
    if not all_results:
        return
    labels, aurocs = [], []
    for r in all_results:
        labels.append(f"{TARGET_DISPLAY.get(r['target'], r['target'])}\n{r['model_name']}")
        aurocs.append(r["test_metrics"].get("AUROC", 0.0))
    colors = ["#2E75B6" if "Gated" in l else "#375623" for l in labels]
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.3), 5))
    bars = ax.bar(range(len(aurocs)), aurocs, color=colors)
    ax.axhline(0.5, linestyle="--", color="gray", lw=1, label="Random (0.5)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Locked-Test AUROC"); ax.set_ylim(0, 1.05)
    ax.set_title("Step 08A MIL — Model Comparison Across All Targets")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _ap = argparse.ArgumentParser(add_help=False)
    _ap.add_argument("--force",    action="store_true")
    _ap.add_argument("--no-cache", action="store_true")
    _flags, _ = _ap.parse_known_args()

    cfg = Step08AConfig()
    set_seed(cfg.global_seed)

    for p in [REPORTS_DIR, FIGURES_DIR, TABLES_DIR, METADATA_DIR, LOGS_DIR, CHECKPOINTS_DIR]:
        ensure_dir(p)

    log_file = LOGS_DIR / f"{STEP_NAME}.log"
    logger   = setup_logger(log_file)
    device   = get_device()
    logger.info("Device: %s", device)

    # ── Cache check ───────────────────────────────────────────────────────────
    _cfg_hash = hashlib.md5(json.dumps({
        "seed": cfg.global_seed, "n_epochs": cfg.n_epochs,
        "attention_dim": cfg.attention_dim,
    }, sort_keys=True).encode()).hexdigest()
    _manifest = PROJECT_ROOT / "cache" / "radiomics_bags" / "step07_cache_manifest.json"
    _inputs   = [INPUT_LESION_CSV, INPUT_FEATURE_DICT]
    _outputs  = [TABLES_DIR / "step07_mil_model_results.csv", METADATA_DIR / "step07_mil_summary.json"]
    if not _flags.force and not _flags.no_cache and _is_step_cached(_manifest, _inputs, _cfg_hash, _outputs):
        logger.info("Cache valid — Step 07 outputs unchanged. Skipping (use --force to re-run).")
        return

    # ── Load data ────────────────────────────────────────────────────────
    if not INPUT_LESION_CSV.exists():
        raise FileNotFoundError(f"Missing: {INPUT_LESION_CSV}  — run Steps 01-03 first.")
    lesion_df = pd.read_csv(INPUT_LESION_CSV)
    logger.info("Loaded lesion table: %s", lesion_df.shape)

    # Load patient splits — must be produced by step_04_feature_engineering.py
    if not INPUT_SPLITS_CSV.exists():
        raise FileNotFoundError(
            f"Splits file not found: {INPUT_SPLITS_CSV}\n"
            f"Run step_04_feature_engineering.py first to generate patient-level splits."
        )
    splits_df = pd.read_csv(INPUT_SPLITS_CSV)

    # Verify every patient in the lesion data has a split assignment
    data_patients  = set(lesion_df["patient_base"].dropna().unique())
    split_patients = set(splits_df["patient_base"].dropna().unique())
    unassigned = data_patients - split_patients
    if unassigned:
        raise ValueError(
            f"Split integrity error: {len(unassigned)} patient(s) in the lesion data have no "
            f"split assignment in {INPUT_SPLITS_CSV}.\n"
            f"Re-run step_04_feature_engineering.py to regenerate a consistent splits file.\n"
            f"Unassigned patients (first 10): {sorted(unassigned)[:10]}"
        )

    # Add label columns if not present (derive from ER/PR/HER2 if needed)
    for src, bin_col in [("ER", "ER_bin"), ("PR", "PR_bin"), ("HER2", "HER2_bin")]:
        if bin_col not in lesion_df.columns and src in lesion_df.columns:
            lesion_df[bin_col] = lesion_df[src].apply(normalize_label_pm)

    feature_cols = load_feature_cols(lesion_df, INPUT_FEATURE_DICT)
    logger.info("Feature columns: %d", len(feature_cols))

    all_results  = []
    all_test_rows = []

    # ── Loop over targets ────────────────────────────────────────────────
    for target_col in cfg.target_columns:
        label_src = LABEL_SOURCE_COLS.get(target_col, "")
        if label_src not in lesion_df.columns:
            logger.warning("Skipping %s — label column '%s' not found.", target_col, label_src)
            continue

        tname = TARGET_DISPLAY.get(target_col, target_col)
        logger.info("=" * 60)
        logger.info("Target: %s", tname)

        # Build bags (fit preprocessor on train only)
        bags, scaler, imputer = build_bags(
            lesion_df, feature_cols, target_col, splits_df, cfg, fit_preprocessor=True
        )
        train_bags = [b for b in bags if b["split"] == "train"]
        valid_bags = [b for b in bags if b["split"] == "valid"]
        test_bags  = [b for b in bags if b["split"] == "test"]

        logger.info("  Bags — train: %d, valid: %d, test: %d",
                    len(train_bags), len(valid_bags), len(test_bags))
        if not train_bags or not valid_bags or not test_bags:
            logger.warning("  Insufficient bags — skipping %s.", tname)
            continue

        # Each split needs both classes for AUROC-based model selection and reporting.
        for split_name, split_bags in [("train", train_bags), ("valid", valid_bags), ("test", test_bags)]:
            split_classes = sorted(set(int(b["label"]) for b in split_bags))
            if len(split_classes) < 2:
                logger.warning("  %s split has one class only for %s: %s — skipping target.", split_name, tname, split_classes)
                train_bags = valid_bags = test_bags = []
                break
        if not train_bags:
            continue

        n_pos_tr = sum(b["label"] for b in train_bags)
        logger.info("  Train label balance: %d pos / %d neg",
                    n_pos_tr, len(train_bags) - n_pos_tr)

        feature_dim = len(feature_cols)

        # ── Train both MIL architectures ──────────────────────────────
        for model_class, model_name in [
            (AttentionMIL,      "AttentionMIL"),
            (GatedAttentionMIL, "GatedAttentionMIL"),
        ]:
            logger.info("  Training %s ...", model_name)
            set_seed(cfg.global_seed)
            result = train_mil(
                model_class, train_bags, valid_bags, test_bags,
                feature_dim, cfg, device, logger, model_name
            )
            result["target"]      = target_col
            result["target_name"] = tname
            result["n_train"]     = len(train_bags)
            result["n_valid"]     = len(valid_bags)
            result["n_test"]      = len(test_bags)
            result["n_features"]  = feature_dim
            all_results.append(result)

            # Log test results
            tm = result["test_metrics"]
            degen_flag = "⚠️ DEGENERATE" if tm.get("degenerate_prediction") else "✓ OK"
            logger.info(
                "  [%s | %s] %s | Test AUROC=%.4f (CI: %.3f–%.3f) | "
                "AUPRC=%.4f | F1=%.4f | BA=%.4f | Sens=%.3f | Spec=%.3f",
                tname, model_name, degen_flag,
                tm.get("AUROC", float("nan")),
                tm.get("AUROC_CI95_lower", float("nan")),
                tm.get("AUROC_CI95_upper", float("nan")),
                tm.get("AUPRC", float("nan")),
                tm.get("F1",    float("nan")),
                tm.get("Balanced_Acc", float("nan")),
                tm.get("Sensitivity",  float("nan")),
                tm.get("Specificity",  float("nan")),
            )
            if tm.get("degenerate_prediction"):
                logger.warning(
                    "  ⚠️  [%s | %s] Degenerate prediction detected "
                    "(Sens=%.2f, Spec=%.2f). Consider increasing n_epochs "
                    "or adjusting min_valid_specificity.",
                    tname, model_name,
                    tm.get("Sensitivity", 0.0), tm.get("Specificity", 0.0)
                )

            # ── Save per-target outputs ──────────────────────────────
            sfx = f"{target_col}_{model_name}"
            if cfg.save_plots:
                plot_training_curves(
                    result["history"],
                    FIGURES_DIR / f"step07_{sfx}_training.png",
                    f"{tname} | {model_name} — Training Curves"
                )
                if len(np.unique(result["test_labels"])) > 1:
                    plot_roc_pr(
                        result["test_labels"], result["test_probs"],
                        FIGURES_DIR / f"step07_{sfx}_roc_pr.png",
                        f"{tname} | {model_name} — Locked Test"
                    )
                # Attention analysis on all bags
                all_bags = train_bags + valid_bags + test_bags
                attn_df  = get_attention_weights(result["model"], all_bags, device)
                plot_attention_distribution(
                    attn_df,
                    FIGURES_DIR / f"step07_{sfx}_attention.png",
                    f"{tname} | {model_name} — Attention Distribution"
                )
                attn_df.to_csv(
                    TABLES_DIR / f"step07_{sfx}_attention_weights.csv",
                    index=False, encoding="utf-8-sig"
                )

            # History CSV
            pd.DataFrame(result["history"]).to_csv(
                TABLES_DIR / f"step07_{sfx}_training_history.csv",
                index=False, encoding="utf-8-sig"
            )

            # Collect test row for master table
            row = {"target": target_col, "target_name": tname, "model": model_name,
                   "n_train": len(train_bags), "n_valid": len(valid_bags), "n_test": len(test_bags),
                   "n_features": feature_dim, "best_threshold": result["best_threshold"],
                   "valid_AUROC":           result["best_valid_auroc"],       # real AUROC at best checkpoint
                   "valid_composite_score": result["best_composite_score"],  # composite used for early stopping
                   **{f"test_{k}": v for k, v in tm.items()}}
            all_test_rows.append(row)

    # ── Master tables ────────────────────────────────────────────────────
    if not all_test_rows:
        logger.error("No results generated. Check input data and labels.")
        return

    master_df = pd.DataFrame(all_test_rows)
    master_csv = TABLES_DIR / "step07_mil_model_results.csv"
    master_df.to_csv(master_csv, index=False, encoding="utf-8-sig")

    # Select the best model using validation-only criteria.
    # The locked test set is reported but never used for model selection.
    best_df = (master_df.sort_values(
                   ["target", "valid_composite_score", "valid_AUROC"],
                   ascending=[True, False, False]
               )
               .groupby("target", as_index=False)
               .head(1)
               .reset_index(drop=True))
    best_df.to_csv(TABLES_DIR / "step07_mil_best_by_target.csv", index=False, encoding="utf-8-sig")

    # Audit-only table: which model would have looked best on test.
    # This is NOT used for model selection, but is useful for detecting test-set volatility.
    best_test_audit_df = (master_df.sort_values(["target", "test_AUROC"], ascending=[True, False])
                          .groupby("target", as_index=False).head(1).reset_index(drop=True))
    best_test_audit_df.to_csv(TABLES_DIR / "step07_mil_best_by_target_test_audit_only.csv",
                              index=False, encoding="utf-8-sig")

    if cfg.save_plots:
        plot_summary_comparison(all_results, FIGURES_DIR / "step07_mil_comparison.png")

    # ── Summary JSON ─────────────────────────────────────────────────────
    summary = {
        "config":         asdict(cfg),
        "device":         str(device),
        "n_patients":     lesion_df["patient_base"].nunique(),
        "n_lesion_rows":  len(lesion_df),
        "n_features":     len(feature_cols),
        "architecture":   "Attention-MIL + Gated-Attention-MIL (Ilse et al. 2018)",
        "selection_policy": "Best model per target is selected by validation composite score, not by locked-test AUROC.",
        "results":        [
            {k: v for k, v in r.items()
             if k not in ("model", "test_probs", "test_labels", "valid_probs", "valid_labels", "history")}
            for r in all_results
        ],
        "best_by_target": best_df.to_dict(orient="records"),
    }
    save_json(summary, METADATA_DIR / "step07_mil_summary.json")

    # ── Publication ZIP ──────────────────────────────────────────────────
    if cfg.save_package:
        zip_path = REPORTS_DIR / "step07_mil_publication_package.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(TABLES_DIR.glob("step07_*")):
                zf.write(p, f"tables/{p.name}")
            for p in sorted(FIGURES_DIR.glob("step07_*")):
                zf.write(p, f"figures/{p.name}")
        logger.info("Publication package: %s", zip_path)

    _save_step_manifest(_manifest, _inputs, _cfg_hash)
    logger.info("=" * 70)
    logger.info("Step 08A completed. Best validation-selected results:")
    for _, row in best_df.iterrows():
        logger.info("  %s | %s | Test AUROC=%.4f | AUPRC=%.4f",
                    row["target_name"], row["model"],
                    row.get("test_AUROC", float("nan")),
                    row.get("test_AUPRC", float("nan")))
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
