"""
Step 08C — Hybrid Fusion: MIL Radiomics + CNN Deep Features
=============================================================
Dataset: 165 patients, RTX 4050

Why Hybrid Fusion?
  - Pure radiomics (Step 05): HER2=0.82, PR=0.67, ER=0.55
  - Step 08A adds learned attention over lesions (MIL)
  - Step 08B adds visual texture from raw MRI (2.5D CNN)
  - This step FUSES both streams → complementary information
  - This step now benchmarks three variants: radiomics_only, image_only, and hybrid_fusion
  - The selected Step 09 model is chosen by validation composite score only

Architecture (Late Fusion):
  Stream A: Radiomics MIL (Attention-MIL on 107 radiomic features)
  Stream B: Image stream  (2.5D CNN embeddings per lesion)
  Fusion:   Concatenate bag-level embeddings → joint classifier
  Both streams share the same patient-level attention pooling concept.

Strategy:
  1. Build radiomics bag embeddings (from Step 08A architecture)
  2. Build image bag embeddings (from Step 08B architecture)
  3. Train radiomics-only, image-only, and hybrid variants
  4. Select the best variant by validation composite score, never by test AUROC

Inputs:
  data/processed/analysis_ready_step03_lesion_only.csv
  metadata/step04_patient_splits.csv
  metadata/step04_feature_dictionary.json

Outputs:
  reports/tables/step09_hybrid_*.csv
  reports/figures/step09_hybrid_*.png
  reports/tables/step09_final_comparison_all_stages.csv
  metadata/step09_hybrid_summary.json
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
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, confusion_matrix, matthews_corrcoef
)
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    import torchvision.models as tv_models
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    raise RuntimeError("Install: pip install torch torchvision")

NIFTI_BACKEND = None
try:
    import nibabel as nib; NIFTI_BACKEND = "nibabel"
except ImportError:
    try:
        import SimpleITK as sitk; NIFTI_BACKEND = "sitk"
    except ImportError:
        pass  # Image stream disabled if no NIfTI library

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("./bcbm_project").resolve()
STEP_NAME    = "step_09_hybrid_fusion"

INPUT_LESION_CSV   = PROJECT_ROOT / "data" / "processed" / "analysis_ready_step03_lesion_only.csv"
INPUT_FEATURE_DICT = PROJECT_ROOT / "metadata" / "step04_feature_dictionary.json"
INPUT_SPLITS_CSV   = PROJECT_ROOT / "metadata" / "step04_patient_splits.csv"

REPORTS_DIR     = PROJECT_ROOT / "reports"
FIGURES_DIR     = REPORTS_DIR  / "figures"
TABLES_DIR      = REPORTS_DIR  / "tables"
METADATA_DIR    = PROJECT_ROOT / "metadata"
LOGS_DIR        = PROJECT_ROOT / "logs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# Shared cache written by Step 08 and read by Step 09.
STEP08_EMBEDDING_CACHE_NPZ = METADATA_DIR / "step08_cnn_embeddings.npz"
STEP08_EMBEDDING_CACHE_INDEX_CSV = METADATA_DIR / "step08_cnn_embeddings_index.csv"
STEP09_EMBEDDING_CACHE_NPZ = METADATA_DIR / "step09_cnn_embedding_cache.npz"
STEP09_EMBEDDING_CACHE_INDEX_CSV = METADATA_DIR / "step09_cnn_embedding_cache_index.csv"

TARGET_COLUMNS = ["target_er", "target_pr", "target_her2"]
LABEL_BINS     = {"target_er": "ER_bin", "target_pr": "PR_bin", "target_her2": "HER2_bin"}
TARGET_DISPLAY = {"target_er": "ER",     "target_pr": "PR",     "target_her2": "HER2"}
GLOBAL_SEED    = 42
RAD_PREFIXES   = ("original_", "wavelet_", "log_sigma_", "shape_",
                  "firstorder_", "glcm_", "glrlm_", "glszm_", "gldm_", "ngtdm_")


# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Step08CConfig:
    project_root:      str   = str(PROJECT_ROOT)
    step_name:         str   = STEP_NAME
    target_columns:    List[str] = field(default_factory=lambda: TARGET_COLUMNS.copy())
    global_seed:       int   = GLOBAL_SEED
    # Radiomics stream
    rad_hidden:        int   = 128
    rad_attn_dim:      int   = 64
    rad_dropout:       float = 0.3
    rad_noise_std:     float = 0.02
    max_rad_instances: int   = 30
    # Image stream
    use_image_stream:  bool  = True    # set False if NIfTI files not available
    image_size:        int   = 64
    cnn_backbone:      str   = "efficientnet_b0"
    cnn_pretrained:    bool  = True
    max_img_instances: int   = 6       # fewer due to GPU memory
    # Fusion
    fusion_hidden:     int   = 128
    fusion_dropout:    float = 0.4
    # Training
    n_epochs:          int   = 100
    lr:                float = 5e-4
    weight_decay:      float = 1e-4
    patience:          int   = 35
    min_valid_specificity: float = 0.20
    min_valid_sensitivity: float = 0.20
    batch_size:        int   = 8
    use_weighted_sampler: bool = True
    use_focal_loss:    bool = False
    focal_gamma:       float = 1.0
    label_smoothing:   float = 0.0
    use_amp:           bool = False
    grad_clip_norm:    float = 0.5
    skip_nonfinite_batches: bool = True
    fusion_variants: List[str] = field(default_factory=lambda: ["radiomics_only", "image_only", "hybrid_fusion"])
    # Eval
    threshold_grid: List[float] = field(default_factory=lambda: [0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65])
    n_bootstrap:       int   = 1000
    save_plots:        bool  = True
    save_package:      bool  = True
    use_embedding_cache: bool = True
    # Prefer Step 08 cache, but keep Step 09 aliases for backward compatibility.
    step08_embedding_cache_npz: str = str(STEP08_EMBEDDING_CACHE_NPZ)
    step08_embedding_cache_index_csv: str = str(STEP08_EMBEDDING_CACHE_INDEX_CSV)
    embedding_cache_npz: str = str(STEP09_EMBEDDING_CACHE_NPZ)
    embedding_cache_index_csv: str = str(STEP09_EMBEDDING_CACHE_INDEX_CSV)


# ════════════════════════════════════════════════════════════════════════════
# UTILITIES  (copied for self-contained file)
# ════════════════════════════════════════════════════════════════════════════

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(p): p.mkdir(parents=True, exist_ok=True)
def save_json(data, path): path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
def load_json(path): return json.loads(path.read_text(encoding="utf-8"))
def get_device(): return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def make_grad_scaler(enabled: bool):
    """Create a GradScaler without deprecated torch.cuda.amp warnings when possible."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled: bool):
    """Create an autocast context without deprecated torch.cuda.amp warnings when possible."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.autocast(device_type="cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


class FocalBCEWithLogitsLoss(nn.Module):
    """Numerically stable binary focal loss with logits and optional label smoothing."""
    def __init__(self, pos_weight: Optional[torch.Tensor] = None, gamma: float = 1.0,
                 label_smoothing: float = 0.0, logit_clip: float = 20.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.logit_clip = float(logit_clip)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=self.logit_clip, neginf=-self.logit_clip)
        logits = logits.clamp(-self.logit_clip, self.logit_clip)
        targets = targets.float()
        if self.label_smoothing > 0:
            eps = float(self.label_smoothing)
            targets = targets * (1.0 - eps) + 0.5 * eps
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal = (1.0 - pt).clamp(0.0, 1.0).pow(self.gamma)
        loss = (focal * bce).mean()
        return torch.nan_to_num(loss, nan=0.0, posinf=50.0, neginf=0.0)


def build_weighted_sampler(records: List[Dict]) -> Optional[WeightedRandomSampler]:
    """Patient-level balanced sampler for small imbalanced train splits."""
    labels = np.asarray([int(r["label"]) for r in records], dtype=int)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return None
    counts = np.bincount(labels, minlength=2).astype(float)
    counts[counts == 0] = 1.0
    class_weights = 1.0 / counts
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

def normalize_label_pm(x):
    if pd.isna(x): return None
    s = str(x).strip()
    if s in ("+","1","1.0"): return 1
    if s in ("-","0","0.0"): return 0
    return None

def setup_logger(log_file):
    logger = logging.getLogger(STEP_NAME)
    logger.setLevel(logging.INFO)
    if logger.handlers: return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, encoding="utf-8"); sh = logging.StreamHandler()
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    from datetime import datetime
    logger.info("=" * 70)
    logger.info("NEW SESSION — %s", datetime.utcnow().isoformat())
    logger.info("=" * 70)
    return logger

def compute_metrics(y_true, probs, preds):
    """Full metric set matching Steps 08A and 08B exactly."""
    if len(np.unique(y_true)) < 2:
        return {"AUROC": float("nan"), "AUPRC": float("nan"), "F1": float("nan"),
                "Balanced_Acc": float("nan")}
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    sens = float(tp / max(tp + fn, 1))
    spec = float(tn / max(tn + fp, 1))
    is_degenerate = (spec < 0.05) or (sens < 0.05)
    return {
        "AUROC":               float(roc_auc_score(y_true, probs)),
        "AUPRC":               float(average_precision_score(y_true, probs)),
        "F1":                  float(f1_score(y_true, preds, zero_division=0)),
        "Balanced_Acc":        float(balanced_accuracy_score(y_true, preds)),
        "Sensitivity":         sens,
        "Specificity":         spec,
        "PPV_Precision":       float(tp / max(tp + fp, 1)),
        "NPV":                 float(tn / max(tn + fn, 1)),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "degenerate_prediction": bool(is_degenerate),
        "MCC":                 float(matthews_corrcoef(y_true, preds)),
    }

def bootstrap_auroc(y_true, probs, n=1000, seed=42):
    rng = np.random.default_rng(seed); aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true)); yb, pb = y_true[idx], probs[idx]
        if len(np.unique(yb)) < 2: continue
        aucs.append(roc_auc_score(yb, pb))
    if len(aucs) < 100: return float("nan"), float("nan")
    return float(np.percentile(aucs,2.5)), float(np.percentile(aucs,97.5))

def choose_threshold(y_true, probs, grid, min_specificity: float = 0.20):
    """
    Choose threshold using Balanced Accuracy with minimum specificity constraint.
    Prevents degenerate solutions where the model predicts everything as positive
    (Sensitivity=1.0, Specificity=0.0). Falls back to unconstrained best if needed.
    Matches the same logic used in Steps 08A and 08B.
    """
    best_t,    best_ba  = 0.5, -1.0   # constrained best
    best_t_fb, best_fb  = 0.5, -1.0   # fallback: unconstrained best

    for t in grid:
        preds = (probs >= t).astype(int)
        ba    = balanced_accuracy_score(y_true, preds)
        tn    = int(((y_true == 0) & (preds == 0)).sum())
        fp    = int(((y_true == 0) & (preds == 1)).sum())
        spec  = tn / max(tn + fp, 1)

        if ba > best_fb:
            best_fb, best_t_fb = ba, t          # track unconstrained best

        if ba > best_ba and spec >= min_specificity:
            best_ba, best_t = ba, t             # track constrained best

    return best_t if best_ba >= 0 else best_t_fb  # fallback if no threshold met constraint


# ════════════════════════════════════════════════════════════════════════════
# IMAGE UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def load_volume_safe(path: str) -> Optional[np.ndarray]:
    if not path or not Path(path).exists(): return None
    try:
        if NIFTI_BACKEND == "nibabel":
            arr = np.asarray(nib.load(path).get_fdata(), dtype=np.float32)
        else:
            arr = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)
        return arr
    except Exception:
        return None

def extract_2p5d(img_path, mask_path, size=64) -> Optional[np.ndarray]:
    vol  = load_volume_safe(img_path)
    mask = load_volume_safe(mask_path)
    if vol is None or mask is None: return None
    mb = (mask > 0.5).astype(np.uint8)
    if mb.sum() == 0: return None
    coords   = np.argwhere(mb)
    centroid = coords.mean(axis=0).astype(int)
    cx, cy, cz = int(centroid[0]), int(centroid[1]), int(centroid[2])
    roi = vol[mb > 0]; mu = roi.mean(); std = roi.std() + 1e-8
    vol = np.clip((vol - mu) / std, -3, 3).astype(np.float32)
    planes = []
    for ax, idx in [(0, cx), (1, cy), (2, cz)]:
        sl = np.take(vol, min(max(idx, 0), vol.shape[ax]-1), axis=ax)
        h, w = sl.shape
        rs = np.linspace(0, h-1, size).astype(int)
        cs = np.linspace(0, w-1, size).astype(int)
        planes.append(sl[np.ix_(rs, cs)])
    result = np.stack(planes, axis=0).astype(np.float32)  # (3, H, W)
    for c in range(3):
        mn, mx = result[c].min(), result[c].max()
        if mx > mn: result[c] = (result[c] - mn) / (mx - mn)
    return result


# ════════════════════════════════════════════════════════════════════════════
# ATTENTION POOLING BLOCK (shared by both streams)
# ════════════════════════════════════════════════════════════════════════════

class GatedAttentionPool(nn.Module):
    """Gated attention pooling: (B, N, D) → (B, D)"""
    def __init__(self, dim: int, attn_dim: int):
        super().__init__()
        self.V = nn.Linear(dim, attn_dim)
        self.U = nn.Linear(dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        A = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)
        A = A.masked_fill(~mask, float("-inf"))
        A = F.softmax(A, dim=1)
        A = torch.nan_to_num(A, nan=0.0)
        return (A.unsqueeze(-1) * h).sum(dim=1), A


# ════════════════════════════════════════════════════════════════════════════
# RADIOMICS STREAM
# ════════════════════════════════════════════════════════════════════════════

class RadiomicsStream(nn.Module):
    def __init__(self, input_dim: int, cfg: Step08CConfig):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, cfg.rad_hidden),
            nn.LayerNorm(cfg.rad_hidden), nn.GELU(), nn.Dropout(cfg.rad_dropout),
            nn.Linear(cfg.rad_hidden, cfg.rad_hidden),
            nn.LayerNorm(cfg.rad_hidden), nn.GELU(), nn.Dropout(cfg.rad_dropout),
        )
        self.pool = GatedAttentionPool(cfg.rad_hidden, cfg.rad_attn_dim)
        self.out_dim = cfg.rad_hidden

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        B, N, D = x.shape
        h = self.encoder(x.view(B*N, D)).view(B, N, -1)
        return self.pool(h, mask)  # (bag_emb, attn_weights)


# ════════════════════════════════════════════════════════════════════════════
# IMAGE STREAM
# ════════════════════════════════════════════════════════════════════════════

class ImageStream(nn.Module):
    def __init__(self, cfg: Step08CConfig, cnn_out_dim: int):
        super().__init__()
        self.pool = GatedAttentionPool(cnn_out_dim, cfg.rad_attn_dim)
        self.out_dim = cnn_out_dim

    def forward(self, img_feats: torch.Tensor, mask: torch.Tensor):
        """img_feats: (B, N, cnn_out_dim) — pre-extracted by backbone"""
        return self.pool(img_feats, mask)


# ════════════════════════════════════════════════════════════════════════════
# HYBRID FUSION MODEL
# ════════════════════════════════════════════════════════════════════════════


class HybridFusionModel(nn.Module):
    """
    Flexible MIL model for three Step 09 variants:
    - radiomics_only: radiomics stream only
    - image_only: CNN embedding stream only
    - hybrid_fusion: radiomics + CNN embeddings
    """
    def __init__(self, rad_stream: Optional[RadiomicsStream],
                 img_stream: Optional[ImageStream],
                 cfg: Step08CConfig,
                 variant: str = "hybrid_fusion"):
        super().__init__()
        if variant not in {"radiomics_only", "image_only", "hybrid_fusion"}:
            raise ValueError(f"Unsupported Step09 variant: {variant}")
        self.variant = variant
        self.rad_stream = rad_stream if variant in {"radiomics_only", "hybrid_fusion"} else None
        self.img_stream = img_stream if variant in {"image_only", "hybrid_fusion"} else None
        self.use_rad = self.rad_stream is not None
        self.use_image = self.img_stream is not None
        if not self.use_rad and not self.use_image:
            raise ValueError(f"Variant {variant} has no active stream.")

        fuse_dim = 0
        if self.use_rad:
            fuse_dim += self.rad_stream.out_dim
        if self.use_image:
            fuse_dim += self.img_stream.out_dim

        hidden = int(getattr(cfg, "fusion_hidden", 64))
        hidden2 = max(hidden // 2, 8)
        drop = float(getattr(cfg, "fusion_dropout", 0.35))
        self.fusion = nn.Sequential(
            nn.LayerNorm(fuse_dim),
            nn.Linear(fuse_dim, hidden),
            nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden2),
            nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden2, 1),
        )

    def forward(self, rad_x, rad_mask, img_feats=None, img_mask=None):
        parts = []
        rad_attn = None
        img_attn = None
        if self.use_rad:
            rad_emb, rad_attn = self.rad_stream(rad_x, rad_mask)
            parts.append(rad_emb)
        if self.use_image:
            if img_feats is None or img_mask is None:
                raise ValueError("Image stream requested but img_feats/img_mask is missing.")
            img_emb, img_attn = self.img_stream(img_feats, img_mask)
            parts.append(img_emb)
        fused = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        logit = self.fusion(fused).squeeze(-1)
        return logit, rad_attn, img_attn


# ════════════════════════════════════════════════════════════════════════════
# DATASET
# ════════════════════════════════════════════════════════════════════════════

class HybridBagDataset(Dataset):
    def __init__(self, records: List[Dict], augment: bool = False, noise_std: float = 0.02):
        self.records   = records
        self.augment   = augment
        self.noise_std = noise_std

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        r      = self.records[idx]
        rad    = torch.from_numpy(r["rad_feats"].copy())
        label  = torch.tensor(r["label"], dtype=torch.float32)
        if self.augment and self.noise_std > 0:
            rad = rad + torch.randn_like(rad) * self.noise_std
        img_tensor = None
        if r.get("img_feats") is not None:
            img_tensor = torch.from_numpy(r["img_feats"].copy())
        return rad, label, img_tensor, r.get("patient_id", "")


def collate_hybrid(batch):
    rads, labels, imgs, patient_ids = zip(*batch)
    max_n_rad = max(r.shape[0] for r in rads)
    d_rad     = rads[0].shape[1]
    B         = len(rads)

    pad_rad   = torch.zeros(B, max_n_rad, d_rad)
    mask_rad  = torch.zeros(B, max_n_rad, dtype=torch.bool)
    for i, r in enumerate(rads):
        n = r.shape[0]
        pad_rad[i, :n] = r
        mask_rad[i, :n] = True

    labels = torch.stack(labels)

    has_img = any(im is not None for im in imgs)
    if has_img:
        d_img = next(im for im in imgs if im is not None).shape[1]
        max_n_img = max((im.shape[0] if im is not None else 0) for im in imgs)
        max_n_img = max(max_n_img, 1)
        pad_img = torch.zeros(B, max_n_img, d_img)
        mask_img = torch.zeros(B, max_n_img, dtype=torch.bool)
        for i, im in enumerate(imgs):
            if im is not None:
                n = im.shape[0]
                pad_img[i, :n] = im
                mask_img[i, :n] = True
    else:
        pad_img = mask_img = None

    return pad_rad, mask_rad, labels, pad_img, mask_img, list(patient_ids)


# ════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION + CNN EMBEDDING CACHE
# ════════════════════════════════════════════════════════════════════════════

def load_feature_cols(lesion_df, feature_dict_path):
    if feature_dict_path.exists():
        fd = load_json(feature_dict_path)
        cols = [c for c in fd.get("numeric_feature_columns_no_scanner_covariates", []) if c in lesion_df.columns]
        if cols:
            return cols
    return [c for c in lesion_df.columns
            if pd.api.types.is_numeric_dtype(lesion_df[c])
            and any(c.startswith(p) for p in RAD_PREFIXES)]


def _cache_signature(cfg: Step08CConfig, lesion_df: pd.DataFrame) -> str:
    return f"{getattr(cfg,'cnn_backbone','efficientnet_b0')}|pre={getattr(cfg,'cnn_pretrained',True)}|size={cfg.image_size}|rows={len(lesion_df)}"


def load_embedding_cache(cfg: Step08CConfig, expected_signature: str, logger=None) -> Dict[int, np.ndarray]:
    """Load Step 08 cache first, then Step 09 compatibility cache if needed."""
    candidates = [
        (
            Path(getattr(cfg, "step08_embedding_cache_npz", STEP08_EMBEDDING_CACHE_NPZ)),
            Path(getattr(cfg, "step08_embedding_cache_index_csv", STEP08_EMBEDDING_CACHE_INDEX_CSV)),
            "Step 08",
        ),
        (
            Path(getattr(cfg, "embedding_cache_npz", STEP09_EMBEDDING_CACHE_NPZ)),
            Path(getattr(cfg, "embedding_cache_index_csv", STEP09_EMBEDDING_CACHE_INDEX_CSV)),
            "Step 09",
        ),
    ]
    for npz_path, idx_path, label in candidates:
        if not (npz_path.exists() and idx_path.exists()):
            continue
        try:
            idx_df = pd.read_csv(idx_path)
            if "signature" in idx_df.columns and idx_df["signature"].dropna().nunique() == 1:
                sig = str(idx_df["signature"].dropna().iloc[0])
                if sig != expected_signature:
                    if logger:
                        logger.info("  %s CNN cache signature mismatch; trying next cache.", label)
                    continue
            data = np.load(npz_path)
            out: Dict[int, np.ndarray] = {}
            for _, row in idx_df.iterrows():
                key = str(row["cache_key"])
                row_index = int(row["row_index"])
                if key in data.files:
                    out[row_index] = np.nan_to_num(data[key], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            if out:
                if logger:
                    logger.info("  Loaded %s CNN embedding cache: %d embeddings from %s", label, len(out), npz_path)
                return out
        except Exception as e:
            if logger:
                logger.warning("  Could not load %s CNN embedding cache (%s); trying next cache.", label, e)
    return {}


def save_embedding_cache(cache: Dict[int, np.ndarray], cfg: Step08CConfig, signature: str, logger=None) -> None:
    if not cache:
        return
    targets = [
        (
            Path(getattr(cfg, "embedding_cache_npz", STEP09_EMBEDDING_CACHE_NPZ)),
            Path(getattr(cfg, "embedding_cache_index_csv", STEP09_EMBEDDING_CACHE_INDEX_CSV)),
        )
    ]
    # If Step 08 cache does not exist, also write the canonical Step 08 cache name
    # so later runs can reuse the same file without re-extraction.
    step08_npz = Path(getattr(cfg, "step08_embedding_cache_npz", STEP08_EMBEDDING_CACHE_NPZ))
    step08_idx = Path(getattr(cfg, "step08_embedding_cache_index_csv", STEP08_EMBEDDING_CACHE_INDEX_CSV))
    if not (step08_npz.exists() and step08_idx.exists()):
        targets.append((step08_npz, step08_idx))

    arrays = {}
    rows = []
    for n, (row_index, emb) in enumerate(sorted(cache.items())):
        key = f"emb_{n:06d}"
        arrays[key] = np.asarray(emb, dtype=np.float32)
        rows.append({"row_index": int(row_index), "cache_key": key, "signature": signature})
    for npz_path, idx_path in targets:
        ensure_dir(npz_path.parent)
        np.savez_compressed(npz_path, **arrays)
        pd.DataFrame(rows).to_csv(idx_path, index=False, encoding="utf-8-sig")
    if logger:
        logger.info("  Saved CNN embedding cache: %d embeddings", len(rows))


def build_or_load_image_embedding_cache(lesion_df, cfg, backbone, device, logger=None) -> Dict[int, np.ndarray]:
    if backbone is None or not getattr(cfg, "use_image_stream", True):
        return {}
    signature = _cache_signature(cfg, lesion_df)
    if getattr(cfg, "use_embedding_cache", True):
        cache = load_embedding_cache(cfg, signature, logger=logger)
        if cache:
            return cache

    img_col  = next((c for c in ["image_abs_path", "image_path"] if c in lesion_df.columns), None)
    mask_col = next((c for c in ["mask_abs_path", "mask_path"] if c in lesion_df.columns), None)
    if img_col is None or mask_col is None:
        if logger: logger.warning("  Image columns not found; image variants disabled.")
        return {}

    cache: Dict[int, np.ndarray] = {}
    n_ok = n_fail = 0
    backbone.eval()
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    with torch.no_grad():
        for idx, row in lesion_df.iterrows():
            ip = str(row[img_col]) if pd.notna(row[img_col]) else ""
            mp = str(row[mask_col]) if pd.notna(row[mask_col]) else ""
            if not ip or not mp:
                n_fail += 1
                continue
            sl = extract_2p5d(ip, mp, size=cfg.image_size)
            if sl is None:
                n_fail += 1
                continue
            sl_hwc = np.transpose(sl, (1,2,0))
            sl_u8 = (sl_hwc * 255).clip(0,255).astype(np.uint8)
            from PIL import Image
            t = transform(Image.fromarray(sl_u8)).unsqueeze(0).to(device)
            emb = backbone(t).squeeze(0).cpu().numpy()
            cache[int(idx)] = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            n_ok += 1
    if logger:
        logger.info("  Image stream extraction/cache build: %d ok, %d failed", n_ok, n_fail)
    if getattr(cfg, "use_embedding_cache", True):
        save_embedding_cache(cache, cfg, signature, logger=logger)
    return cache


def build_hybrid_records(
    lesion_df, feature_cols, target_col, splits_df, cfg,
    image_embedding_cache: Optional[Dict[int, np.ndarray]] = None,
    scaler=None, imputer=None, fit_preprocessor=False, logger=None,
):
    label_src = LABEL_BINS.get(target_col, "")
    df = lesion_df.copy().merge(splits_df[["patient_base","split"]], on="patient_base", how="left")
    df = df[df["split"].isin(["train", "valid", "test"])].copy()
    df["_lbl"] = df[label_src].apply(normalize_label_pm) if label_src in df.columns else None
    df = df[df["_lbl"].notna()].copy()

    X = df[feature_cols].copy().replace([np.inf,-np.inf], np.nan).apply(pd.to_numeric, errors="coerce")
    if fit_preprocessor:
        train_mask = df["split"] == "train"
        if train_mask.sum() == 0:
            raise ValueError(f"No training rows available for {target_col}.")
        imputer = SimpleImputer(strategy="median")
        imputer.fit(X.loc[train_mask])
        X_imp = imputer.transform(X)
        scaler = RobustScaler()
        scaler.fit(X_imp[train_mask.values])
    else:
        if imputer is None or scaler is None:
            raise ValueError("scaler and imputer are required when fit_preprocessor=False")
        X_imp = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    df_feat = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)

    image_embedding_cache = image_embedding_cache or {}
    records = []
    for patient_id, grp in df.groupby("patient_base"):
        label_vals = grp["_lbl"].unique()
        if len(label_vals) > 1:
            continue
        split_vals = grp["split"].dropna().unique()
        if len(split_vals) != 1:
            continue
        label = int(label_vals[0])
        split = str(split_vals[0])

        rad_feats = df_feat.loc[grp.index].values.astype(np.float32)
        rad_feats = np.nan_to_num(rad_feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if len(rad_feats) > cfg.max_rad_instances:
            pick = np.random.choice(len(rad_feats), cfg.max_rad_instances, replace=False)
            rad_feats = rad_feats[pick]

        img_embs = [image_embedding_cache[int(i)] for i in grp.index if int(i) in image_embedding_cache]
        if img_embs and len(img_embs) > cfg.max_img_instances:
            img_embs = img_embs[:cfg.max_img_instances]
        img_arr = np.stack(img_embs, axis=0).astype(np.float32) if img_embs else None
        if img_arr is not None:
            img_arr = np.nan_to_num(img_arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        records.append({
            "patient_id": patient_id,
            "label": label,
            "split": split,
            "rad_feats": rad_feats,
            "img_feats": img_arr,
        })
    return records, scaler, imputer


# ════════════════════════════════════════════════════════════════════════════
# TRAINING ENGINE
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_hybrid(model, loader, device, criterion: Optional[nn.Module] = None):
    model.eval()
    all_probs, all_labels, all_patient_ids = [], [], []
    total_loss = 0.0
    n_samples = 0
    for rad, rad_mask, labels, img, img_mask, patient_ids in loader:
        rad, rad_mask = rad.to(device), rad_mask.to(device)
        labels_dev = labels.to(device)
        if img is not None:
            img, img_mask = img.to(device), img_mask.to(device)
        logits, _, _ = model(rad, rad_mask, img, img_mask)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        if criterion is not None:
            loss_val = criterion(logits, labels_dev)
            if torch.isfinite(loss_val):
                total_loss += float(loss_val.item()) * len(labels)
                n_samples += len(labels)
        all_probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())
        all_patient_ids.extend(patient_ids)
    avg_loss = total_loss / max(n_samples, 1) if criterion is not None else float("nan")
    return np.array(all_probs), np.array(all_labels, dtype=int), avg_loss, all_patient_ids


def compute_composite(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    auroc = roc_auc_score(y_true, probs)
    preds = (probs >= threshold).astype(int)
    ba = balanced_accuracy_score(y_true, preds)
    tn = int(((y_true == 0) & (preds == 0)).sum())
    fp = int(((y_true == 0) & (preds == 1)).sum())
    fn = int(((y_true == 1) & (preds == 0)).sum())
    tp = int(((y_true == 1) & (preds == 1)).sum())
    spec = tn / max(tn + fp, 1)
    sens = tp / max(tp + fn, 1)
    penalty = 0.0
    if spec < 0.10: penalty += 0.10
    if sens < 0.10: penalty += 0.10
    return float(0.7 * auroc + 0.3 * ba - penalty)


def choose_threshold_non_degenerate(y_true, probs, grid, min_specificity=0.20, min_sensitivity=0.20):
    """Validation-only threshold selection that avoids all-positive/all-negative solutions."""
    best_t, best_score = 0.5, -1e9
    best_fb_t, best_fb_score = 0.5, -1e9
    for t in grid:
        preds = (probs >= t).astype(int)
        ba = balanced_accuracy_score(y_true, preds)
        tn = int(((y_true == 0) & (preds == 0)).sum())
        fp = int(((y_true == 0) & (preds == 1)).sum())
        fn = int(((y_true == 1) & (preds == 0)).sum())
        tp = int(((y_true == 1) & (preds == 1)).sum())
        spec = tn / max(tn + fp, 1)
        sens = tp / max(tp + fn, 1)
        score = ba - 0.05 * abs(sens - spec)
        if score > best_fb_score:
            best_fb_score, best_fb_t = score, t
        if spec >= min_specificity and sens >= min_sensitivity and score > best_score:
            best_score, best_t = score, t
    return best_t if best_score > -1e8 else best_fb_t


def make_prediction_df(patient_ids, split, y_true, probs, threshold, target, variant):
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    pred = (probs >= threshold).astype(int)
    return pd.DataFrame({
        "patient_id": patient_ids,
        "split": split,
        "target": target,
        "variant": variant,
        "y_true": y_true,
        "prob": probs,
        "threshold": float(threshold),
        "pred": pred,
    })


def train_hybrid_variant(records, target_col, feature_cols, cfg, device, logger, variant: str):
    tname = TARGET_DISPLAY.get(target_col, target_col)
    train_rec = [r for r in records if r["split"] == "train"]
    valid_rec = [r for r in records if r["split"] == "valid"]
    test_rec  = [r for r in records if r["split"] == "test"]
    if not train_rec or not valid_rec or not test_rec:
        logger.warning("  Insufficient records for %s/%s.", tname, variant)
        return None

    has_image = any(r.get("img_feats") is not None for r in records)
    sample_img = next((r.get("img_feats") for r in records if r.get("img_feats") is not None), None)
    cnn_out_dim = int(sample_img.shape[-1]) if sample_img is not None and sample_img.ndim == 2 else 0
    if variant in {"image_only", "hybrid_fusion"} and (not has_image or cnn_out_dim <= 0):
        logger.warning("  Skipping %s/%s: no usable image embeddings.", tname, variant)
        return None

    input_dim = len(feature_cols)
    rad_stream = RadiomicsStream(input_dim, cfg) if variant in {"radiomics_only", "hybrid_fusion"} else None
    img_stream = ImageStream(cfg, cnn_out_dim) if variant in {"image_only", "hybrid_fusion"} else None
    model = HybridFusionModel(rad_stream, img_stream, cfg, variant=variant).to(device)

    n_pos = sum(r["label"] for r in train_rec)
    n_neg = len(train_rec) - n_pos
    pos_weight_value = float(np.clip(n_neg / max(n_pos, 1), 0.25, 4.0))
    pos_w = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    if getattr(cfg, "use_focal_loss", False):
        criterion = FocalBCEWithLogitsLoss(pos_weight=pos_w, gamma=getattr(cfg, "focal_gamma", 1.0), label_smoothing=getattr(cfg, "label_smoothing", 0.0))
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    optimizer = torch.optim.AdamW(model.parameters(), lr=getattr(cfg, "lr", 1e-4), weight_decay=getattr(cfg, "weight_decay", 5e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=getattr(cfg, "n_epochs", 70))

    kw = dict(batch_size=cfg.batch_size, collate_fn=collate_hybrid, num_workers=0, pin_memory=(device.type == "cuda"))
    train_ds = HybridBagDataset(train_rec, augment=True, noise_std=cfg.rad_noise_std)
    sampler = build_weighted_sampler(train_rec) if getattr(cfg, "use_weighted_sampler", True) else None
    train_loader = DataLoader(train_ds, shuffle=(sampler is None), sampler=sampler, **kw)
    valid_loader = DataLoader(HybridBagDataset(valid_rec, augment=False), shuffle=False, **kw)
    test_loader  = DataLoader(HybridBagDataset(test_rec,  augment=False), shuffle=False, **kw)

    _use_amp = bool(getattr(cfg, "use_amp", False)) and device.type == "cuda"
    scaler_amp = make_grad_scaler(enabled=_use_amp)
    best_composite_score = -1.0
    best_valid_auroc_real = -1.0
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    patience_cnt = 0
    history = []
    skipped_batches = 0

    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        total_loss = 0.0
        seen_samples = 0
        for rad, rad_mask, labels, img, img_mask, _patient_ids in train_loader:
            rad = torch.nan_to_num(rad.to(device), nan=0.0, posinf=0.0, neginf=0.0)
            rad_mask, labels = rad_mask.to(device), labels.to(device)
            if img is not None:
                img = torch.nan_to_num(img.to(device), nan=0.0, posinf=0.0, neginf=0.0)
                img_mask = img_mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(enabled=_use_amp):
                logits, _, _ = model(rad, rad_mask, img, img_mask)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                skipped_batches += 1
                if getattr(cfg, "skip_nonfinite_batches", True):
                    continue
                loss = torch.nan_to_num(loss, nan=0.0, posinf=50.0, neginf=0.0)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), getattr(cfg, "grad_clip_norm", 0.5))
            scaler_amp.step(optimizer)
            scaler_amp.update()
            total_loss += float(loss.detach().item()) * len(labels)
            seen_samples += len(labels)
        scheduler.step()

        v_probs, v_labels, v_loss, _ = eval_hybrid(model, valid_loader, device, criterion=criterion)
        v_probs = np.nan_to_num(v_probs, nan=0.5, posinf=1.0, neginf=0.0)
        v_auroc = roc_auc_score(v_labels, v_probs) if len(np.unique(v_labels)) > 1 else 0.5
        tl = total_loss / max(seen_samples, 1)
        composite = compute_composite(v_labels, v_probs, threshold=0.5)
        history.append({
            "epoch": epoch,
            "variant": variant,
            "train_loss": round(float(tl), 4),
            "valid_loss": round(float(v_loss), 4),
            "valid_auroc": round(float(v_auroc), 4),
            "valid_composite": round(float(composite), 4),
            "skipped_nonfinite_batches": int(skipped_batches),
        })
        if composite > best_composite_score:
            best_composite_score = composite
            best_valid_auroc_real = v_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
        if epoch % 20 == 0:
            logger.info("  [%s] Epoch %3d | loss=%.4f | valid_loss=%.4f | valid_AUROC=%.4f | composite=%.4f (best=%.4f)",
                        variant, epoch, tl, v_loss, v_auroc, composite, best_composite_score)
        if patience_cnt >= cfg.patience:
            logger.info("  [%s] Early stop at epoch %d", variant, epoch)
            break

    if skipped_batches > 0:
        logger.warning("  [%s] Skipped %d non-finite training batch(es) for %s.", variant, skipped_batches, tname)
    model.load_state_dict(best_state)

    v_probs, v_labels, v_loss_final, v_pids = eval_hybrid(model, valid_loader, device, criterion=criterion)
    v_probs = np.nan_to_num(v_probs, nan=0.5, posinf=1.0, neginf=0.0)
    best_thr = choose_threshold_non_degenerate(
        v_labels, v_probs, cfg.threshold_grid,
        min_specificity=getattr(cfg, "min_valid_specificity", 0.20),
        min_sensitivity=getattr(cfg, "min_valid_sensitivity", 0.20),
    )
    valid_mets = compute_metrics(v_labels, v_probs, (v_probs >= best_thr).astype(int))
    valid_mets["Loss"] = float(v_loss_final)

    t_probs, t_labels, t_loss_final, t_pids = eval_hybrid(model, test_loader, device, criterion=criterion)
    t_probs = np.nan_to_num(t_probs, nan=0.5, posinf=1.0, neginf=0.0)
    t_preds = (t_probs >= best_thr).astype(int)
    test_mets = compute_metrics(t_labels, t_probs, t_preds)
    test_mets["Loss"] = float(t_loss_final)
    ci_lo, ci_hi = bootstrap_auroc(t_labels, t_probs, n=cfg.n_bootstrap)
    test_mets["AUROC_CI95_lower"] = ci_lo
    test_mets["AUROC_CI95_upper"] = ci_hi
    test_mets["generalization_gap"] = float(valid_mets["AUROC"] - test_mets["AUROC"])
    test_mets["skipped_nonfinite_batches"] = int(skipped_batches)

    pred_df = pd.concat([
        make_prediction_df(v_pids, "valid", v_labels, v_probs, best_thr, target_col, variant),
        make_prediction_df(t_pids, "test", t_labels, t_probs, best_thr, target_col, variant),
    ], ignore_index=True)

    return {
        "variant": variant,
        "history": history,
        "best_threshold": best_thr,
        "valid_metrics": valid_mets,
        "test_metrics": test_mets,
        "test_probs": t_probs,
        "test_labels": t_labels,
        "prediction_df": pred_df,
        "best_composite_score": best_composite_score,
        "best_valid_auroc": best_valid_auroc_real,
        "n_train": len(train_rec), "n_valid": len(valid_rec), "n_test": len(test_rec),
        "image_stream_used": bool(variant in {"image_only", "hybrid_fusion"} and has_image and getattr(cfg, "use_image_stream", True)),
    }


def select_best_variant(results: List[Dict]) -> Optional[Dict]:
    if not results:
        return None
    # Selection is validation-only. Degenerate validation predictions are penalized but not impossible.
    def key_fn(r):
        vm = r.get("valid_metrics", {})
        deg_penalty = 0.10 if vm.get("degenerate_prediction", False) else 0.0
        return (float(r.get("best_composite_score", -1.0)) - deg_penalty,
                float(r.get("best_valid_auroc", -1.0)))
    return sorted(results, key=key_fn, reverse=True)[0]


# ════════════════════════════════════════════════════════════════════════════
# CROSS-STAGE COMPARISON TABLE
# ════════════════════════════════════════════════════════════════════════════

def build_cross_stage_comparison(step09_rows: List[Dict]) -> pd.DataFrame:
    """Load best results from Steps 05, 07, 08, and selected Step 09 variants."""
    rows = []
    # Step 05 must contribute the prespecified radiomic arm, not the best block per target.
    # step05master_global_best_models_by_target.csv picks the winning feature-set variant across
    # blocks, which for ER and HER2 is acquisition_only -- the scanner control. Putting that in a
    # figure captioned "tabular radiomics" compares the imaging stages against the control while
    # calling it radiomics. Read the radiomics_pure result for each target instead.
    t05 = PROJECT_ROOT / "reports" / "tables"
    for target in ("target_er", "target_pr", "target_her2"):
        s05 = t05 / f"step05master_{target}_radiomics_pure_best_model_test_result.csv"
        if not s05.exists():
            continue
        r = pd.read_csv(s05).iloc[0]
        rows.append({"stage": "Step05_Radiomics", "target": target,
                     "model": r.get("model_display", r.get("model","")),
                     "Test_AUROC": r.get("AUROC", float("nan")),
                     "Test_AUPRC": r.get("AUPRC", float("nan")),
                     "Test_F1":    r.get("F1",    float("nan")),
                     "Test_BA":    r.get("Balanced_Acc", float("nan"))})
    s07 = PROJECT_ROOT / "reports" / "tables" / "step07_mil_best_by_target.csv"
    if s07.exists():
        df07 = pd.read_csv(s07)
        for _, r in df07.iterrows():
            rows.append({"stage": "Step07_MIL", "target": r.get("target",""),
                         "model": r.get("model",""),
                         "Test_AUROC": r.get("test_AUROC", float("nan")),
                         "Test_AUPRC": r.get("test_AUPRC", float("nan")),
                         "Test_F1":    r.get("test_F1",    float("nan")),
                         "Test_BA":    r.get("test_Balanced_Acc", float("nan"))})
    s08 = PROJECT_ROOT / "reports" / "tables" / "step08_cnn_model_results.csv"
    if s08.exists():
        df08 = pd.read_csv(s08)
        for _, r in df08.iterrows():
            rows.append({"stage": "Step08_CNN", "target": r.get("target",""),
                         "model": r.get("backbone",""),
                         "Test_AUROC": r.get("test_AUROC", float("nan")),
                         "Test_AUPRC": r.get("test_AUPRC", float("nan")),
                         "Test_F1":    r.get("test_F1",    float("nan")),
                         "Test_BA":    r.get("test_Balanced_Acc", float("nan"))})
    for r in step09_rows:
        tm = r.get("test_metrics", {})
        # "Step09_Selected" is the variant chosen on validation, which is not always the
        # fusion model: for ER and HER2 validation chose image_only. Any figure drawing this
        # row must say "validation-selected", not "hybrid".
        rows.append({"stage": "Step09_Selected", "target": r.get("target",""),
                     "model": r.get("variant", "Step09"),
                     "Test_AUROC": tm.get("AUROC", float("nan")),
                     "Test_AUPRC": tm.get("AUPRC", float("nan")),
                     "Test_F1":    tm.get("F1",    float("nan")),
                     "Test_BA":    tm.get("Balanced_Acc", float("nan"))})
    return pd.DataFrame(rows)


def plot_comparison(comp_df: pd.DataFrame, path: Path) -> None:
    if comp_df.empty:
        return
    targets = [c for c in ["target_er","target_pr","target_her2"] if c in comp_df["target"].values]
    stages = comp_df["stage"].unique().tolist()
    tdisp = {"target_er":"ER","target_pr":"PR","target_her2":"HER2"}
    colors = ["#1F4E79","#2E75B6","#E36C0A","#375623","#C00000","#7B2C9E"]
    fig, axes = plt.subplots(1, len(targets), figsize=(5.5*len(targets), 5.5), sharey=True)
    if len(targets) == 1:
        axes = [axes]
    for ax, tgt in zip(axes, targets):
        sub = comp_df[comp_df["target"] == tgt]
        x = np.arange(len(stages))
        vals = [sub[sub["stage"] == st]["Test_AUROC"].max() if not sub[sub["stage"] == st].empty else np.nan for st in stages]
        bars = ax.bar(x, vals, color=colors[:len(stages)])
        ax.axhline(0.5, linestyle="--", color="gray", lw=1, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_","\n") for s in stages], fontsize=8, rotation=15, ha="right")
        ax.set_ylim(0, 1.1)
        ax.set_title(tdisp.get(tgt, tgt), fontsize=13, fontweight="bold")
        ax.set_ylabel("Test AUROC")
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x()+bar.get_width()/2, v+0.02, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    plt.suptitle("Cross-Stage Comparison — Steps 05 → 09", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_training(history, path, title):
    if not history:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    epochs = [h["epoch"] for h in history]
    axes[0].plot(epochs, [h["train_loss"] for h in history], color="#2E75B6", label="train")
    axes[0].plot(epochs, [h.get("valid_loss", np.nan) for h in history], color="#C00000", linestyle="--", label="valid")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, [h["valid_auroc"] for h in history], color="#375623", lw=2, label="Real AUROC")
    axes[1].axhline(0.5, linestyle="--", color="gray", alpha=0.5, label="Random")
    axes[1].set_title("Validation AUROC"); axes[1].set_xlabel("Epoch"); axes[1].set_ylim(0,1); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].plot(epochs, [h.get("valid_composite", np.nan) for h in history], color="#7B2C9E", lw=2, label="Composite")
    axes[2].axhline(0.5, linestyle="--", color="gray", alpha=0.5)
    axes[2].set_title("Composite Score"); axes[2].set_xlabel("Epoch"); axes[2].set_ylim(0,1); axes[2].legend(); axes[2].grid(alpha=0.3)
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_roc_pr(y_true, probs, path, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fpr, tpr, _ = roc_curve(y_true, probs)
    ax1.plot(fpr, tpr, lw=2, color="#1F4E79", label=f"AUROC={roc_auc_score(y_true,probs):.3f}")
    ax1.plot([0,1],[0,1],"--",color="gray",lw=1); ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
    ax1.set_title("ROC"); ax1.legend(); ax1.grid(alpha=0.3)
    prec, rec, _ = precision_recall_curve(y_true, probs)
    ax2.plot(rec, prec, lw=2, color="#C00000", label=f"AUPRC={average_precision_score(y_true,probs):.3f}")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision"); ax2.set_title("PR Curve")
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.suptitle(title, fontsize=12); plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _ap = argparse.ArgumentParser(add_help=False)
    _ap.add_argument("--force",    action="store_true")
    _ap.add_argument("--no-cache", action="store_true")
    _flags, _ = _ap.parse_known_args()

    cfg = Step08CConfig()
    set_seed(cfg.global_seed)
    for p in [REPORTS_DIR, FIGURES_DIR, TABLES_DIR, METADATA_DIR, LOGS_DIR, CHECKPOINTS_DIR]:
        ensure_dir(p)
    logger = setup_logger(LOGS_DIR / f"{STEP_NAME}.log")
    device = get_device()
    logger.info("Device: %s | Image stream: %s | NIfTI: %s", device, getattr(cfg, "use_image_stream", True), NIFTI_BACKEND)
    logger.info("Step09 variants: %s | sampler=%s | focal=%s | gamma=%.2f | amp=%s | lr=%.1e | cache=%s",
                getattr(cfg, "fusion_variants", []), getattr(cfg, "use_weighted_sampler", True),
                getattr(cfg, "use_focal_loss", False), getattr(cfg, "focal_gamma", 1.0),
                getattr(cfg, "use_amp", False), getattr(cfg, "lr", 1e-4), getattr(cfg, "use_embedding_cache", True))

    # ── Cache check ───────────────────────────────────────────────────────────
    _cfg_hash = hashlib.md5(json.dumps({
        "backbone": cfg.cnn_backbone, "n_epochs": cfg.n_epochs,
        "seed": cfg.global_seed, "variants": sorted(cfg.fusion_variants),
    }, sort_keys=True).encode()).hexdigest()
    _manifest = PROJECT_ROOT / "cache" / "cnn_embeddings" / "step09_cache_manifest.json"
    _inputs   = [INPUT_LESION_CSV, INPUT_FEATURE_DICT, INPUT_SPLITS_CSV]
    _outputs  = [TABLES_DIR / "step09_hybrid_model_results.csv", METADATA_DIR / "step09_hybrid_summary.json"]
    if not _flags.force and not _flags.no_cache and _is_step_cached(_manifest, _inputs, _cfg_hash, _outputs):
        logger.info("Cache valid — Step 09 outputs unchanged. Skipping (use --force to re-run).")
        return

    if not INPUT_LESION_CSV.exists():
        raise FileNotFoundError(f"Missing: {INPUT_LESION_CSV}")
    lesion_df = pd.read_csv(INPUT_LESION_CSV)
    logger.info("Loaded lesion table: %s", lesion_df.shape)
    if not INPUT_SPLITS_CSV.exists():
        raise FileNotFoundError(f"Splits file not found: {INPUT_SPLITS_CSV}")
    splits_df = pd.read_csv(INPUT_SPLITS_CSV)
    data_patients = set(lesion_df["patient_base"].dropna().unique())
    split_patients = set(splits_df["patient_base"].dropna().unique())
    unassigned = data_patients - split_patients
    if unassigned:
        raise ValueError(f"Split integrity error: {len(unassigned)} patient(s) missing split assignment. First 10: {sorted(unassigned)[:10]}")

    for src, bc in [("ER","ER_bin"),("PR","PR_bin"),("HER2","HER2_bin")]:
        if bc not in lesion_df.columns and src in lesion_df.columns:
            lesion_df[bc] = lesion_df[src].apply(normalize_label_pm)

    feature_cols = load_feature_cols(lesion_df, INPUT_FEATURE_DICT)
    logger.info("Radiomics features: %d", len(feature_cols))

    backbone, cnn_out_dim = None, 0
    if getattr(cfg, "use_image_stream", True) and NIFTI_BACKEND is not None:
        try:
            weights = tv_models.EfficientNet_B0_Weights.DEFAULT if cfg.cnn_pretrained else None
            _bb = tv_models.efficientnet_b0(weights=weights)
            cnn_out_dim = _bb.classifier[1].in_features
            _bb.classifier = nn.Identity()
            backbone = _bb.to(device).eval()
            logger.info("CNN backbone loaded: EfficientNet-B0 | out_dim=%d", cnn_out_dim)
        except Exception as e:
            logger.warning("Could not load CNN backbone (%s) — image variants disabled.", e)
            backbone = None
            cfg.use_image_stream = False
    else:
        logger.info("Image stream disabled (use_image_stream=False or no NIfTI library).")

    image_cache = build_or_load_image_embedding_cache(lesion_df, cfg, backbone, device, logger=logger) if backbone is not None else {}
    if not image_cache:
        cfg.fusion_variants = [v for v in getattr(cfg, "fusion_variants", ["radiomics_only"]) if v == "radiomics_only"]
        logger.warning("No image embeddings available; running radiomics_only only.")

    all_variant_rows = []
    selected_rows = []
    all_predictions = []

    for target_col in cfg.target_columns:
        tname = TARGET_DISPLAY.get(target_col, target_col)
        logger.info("=" * 60)
        logger.info("Target: %s", tname)
        set_seed(cfg.global_seed)
        records, scaler, imputer = build_hybrid_records(
            lesion_df, feature_cols, target_col, splits_df, cfg,
            image_embedding_cache=image_cache,
            fit_preprocessor=True, logger=logger,
        )
        if len(records) < 10:
            logger.warning("  Only %d records — skipping %s.", len(records), tname)
            continue

        target_results = []
        for variant in getattr(cfg, "fusion_variants", ["radiomics_only", "image_only", "hybrid_fusion"]):
            logger.info("  Training variant: %s", variant)
            set_seed(cfg.global_seed)
            result = train_hybrid_variant(records, target_col, feature_cols, cfg, device, logger, variant=variant)
            if result is None:
                continue
            result["target"] = target_col
            target_results.append(result)
            tm = result["test_metrics"]
            vm = result["valid_metrics"]
            flag = "DEGENERATE" if tm.get("degenerate_prediction") else "OK"
            logger.info("  [%s | %s] %s | valid_AUROC=%.4f | valid_comp=%.4f | Test AUROC=%.4f | BA=%.4f | Sens=%.3f | Spec=%.3f | thr=%.2f",
                        tname, variant, flag, result.get("best_valid_auroc", float("nan")),
                        result.get("best_composite_score", float("nan")), tm.get("AUROC", float("nan")),
                        tm.get("Balanced_Acc", float("nan")), tm.get("Sensitivity", float("nan")),
                        tm.get("Specificity", float("nan")), result.get("best_threshold", float("nan")))
            sfx = f"{target_col}_{variant}"
            if cfg.save_plots:
                plot_training(result["history"], FIGURES_DIR / f"step09_{sfx}_training.png", f"{tname} | {variant}")
                if len(np.unique(result["test_labels"])) > 1:
                    plot_roc_pr(result["test_labels"], result["test_probs"], FIGURES_DIR / f"step09_{sfx}_roc_pr.png", f"{tname} | {variant} — Locked Test")
            pd.DataFrame(result["history"]).to_csv(TABLES_DIR / f"step09_{sfx}_history.csv", index=False, encoding="utf-8-sig")
            result["prediction_df"].to_csv(TABLES_DIR / f"step09_{sfx}_predictions.csv", index=False, encoding="utf-8-sig")
            all_predictions.append(result["prediction_df"])

        selected = select_best_variant(target_results)
        if selected is None:
            logger.warning("  No valid variants produced for %s.", tname)
            continue
        logger.info("  Selected Step09 variant for %s: %s (valid_comp=%.4f, valid_AUROC=%.4f)",
                    tname, selected["variant"], selected.get("best_composite_score", float("nan")), selected.get("best_valid_auroc", float("nan")))
        selected_rows.append(selected)
        all_variant_rows.extend(target_results)

    if not all_variant_rows:
        logger.error("No results generated.")
        return

    def row_from_result(r: Dict, selected: bool) -> Dict:
        tm = r["test_metrics"]
        vm = r.get("valid_metrics", {})
        return {
            "target": r["target"],
            "model": "Step09FusionBenchmark",
            "variant": r["variant"],
            "selected_by_validation": bool(selected),
            "image_stream_used": r["image_stream_used"],
            "n_train": r["n_train"], "n_valid": r["n_valid"], "n_test": r["n_test"],
            "valid_AUROC": r["best_valid_auroc"],
            "valid_composite_score": r["best_composite_score"],
            "valid_Sensitivity": vm.get("Sensitivity", float("nan")),
            "valid_Specificity": vm.get("Specificity", float("nan")),
            "best_threshold": r["best_threshold"],
            **{f"test_{k}": v for k, v in tm.items()}
        }

    selected_ids = {(r["target"], r["variant"]) for r in selected_rows}
    master_rows = [row_from_result(r, (r["target"], r["variant"]) in selected_ids) for r in all_variant_rows]
    master_df = pd.DataFrame(master_rows)
    master_df.to_csv(TABLES_DIR / "step09_hybrid_model_results.csv", index=False, encoding="utf-8-sig")
    best_df = pd.DataFrame([row_from_result(r, True) for r in selected_rows])
    best_df.to_csv(TABLES_DIR / "step09_hybrid_best_by_target.csv", index=False, encoding="utf-8-sig")
    if all_predictions:
        pd.concat(all_predictions, ignore_index=True).to_csv(TABLES_DIR / "step09_all_variant_predictions.csv", index=False, encoding="utf-8-sig")

    comp_df = build_cross_stage_comparison(selected_rows)
    comp_df.to_csv(TABLES_DIR / "step09_final_comparison_all_stages.csv", index=False, encoding="utf-8-sig")
    if cfg.save_plots:
        plot_comparison(comp_df, FIGURES_DIR / "step09_final_stage_comparison.png")

    save_json({
        "config": asdict(cfg),
        "device": str(device),
        "nifti_backend": NIFTI_BACKEND,
        "selection_policy": "best Step09 variant per target selected by validation composite score only",
        "results": master_rows,
        "selected_results": [row_from_result(r, True) for r in selected_rows],
    }, METADATA_DIR / "step09_hybrid_summary.json")
    _save_step_manifest(_manifest, _inputs, _cfg_hash)

    if cfg.save_package:
        zip_path = REPORTS_DIR / "step09_hybrid_publication_package.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(TABLES_DIR.glob("step09*")):
                zf.write(p, f"tables/{p.name}")
            for p in sorted(FIGURES_DIR.glob("step09*")):
                zf.write(p, f"figures/{p.name}")
        logger.info("Publication package: %s", zip_path)

    logger.info("=" * 70)
    logger.info("Step 09 (Fusion Benchmark) complete.")
    logger.info("Cross-stage summary saved: %s", TABLES_DIR / "step09_final_comparison_all_stages.csv")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
