"""
Step 08B — Strong 2.5D CNN-MIL Transfer Learning for Receptor Status Prediction
==========================================================================
Dataset: 165 patients, RTX 4050 GPU

Why 2.5D instead of full 3D CNN:
  - 165 patients is far too small for 3D CNN from scratch (needs 1000+)
  - 2.5D extracts 3 orthogonal slices per lesion → treat as 3-channel RGB input
  - Leverages ImageNet-pretrained weights (EfficientNet-B0 / ResNet-18)
  - RTX 4050 has ~6GB VRAM → 3D volumes at 128^3 float32 use ~8GB → too large
  - 2.5D fits comfortably in memory and trains in minutes

Architecture:
  Stage 1 — Per-lesion: 2.5D slice extraction → EfficientNet-B0 feature extractor
  Stage 2 — Per-patient: Attention pooling over all lesion embeddings → classifier
  This is a CNN + MIL hybrid — the best approach for multi-lesion brain MRI.

Input:
  data/processed/analysis_ready_step03_lesion_only.csv  (needs image_abs_path, mask_abs_path)
  metadata/step04_patient_splits.csv

Output:
  reports/tables/step08_cnn_*.csv
  reports/figures/step08_cnn_*.png
  metadata/step08_cnn_summary.json

Requirements:
  pip install torch torchvision nibabel
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

warnings.filterwarnings("ignore", category=UserWarning)

# ── PyTorch ──────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    import torchvision.models as tv_models
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    raise RuntimeError("Install: pip install torch torchvision")

# ── NIfTI ────────────────────────────────────────────────────────────────────
NIFTI_BACKEND = None
try:
    import nibabel as nib
    NIFTI_BACKEND = "nibabel"
except ImportError:
    try:
        import SimpleITK as sitk
        NIFTI_BACKEND = "sitk"
    except ImportError:
        raise RuntimeError("Install nibabel: pip install nibabel --break-system-packages")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("./bcbm_project").resolve()
STEP_NAME    = "step_08_cnn_image"

INPUT_LESION_CSV = PROJECT_ROOT / "data" / "processed" / "analysis_ready_step03_lesion_only.csv"
INPUT_SPLITS_CSV = PROJECT_ROOT / "metadata" / "step04_patient_splits.csv"

REPORTS_DIR     = PROJECT_ROOT / "reports"
FIGURES_DIR     = REPORTS_DIR  / "figures"
TABLES_DIR      = REPORTS_DIR  / "tables"
METADATA_DIR    = PROJECT_ROOT / "metadata"
LOGS_DIR        = PROJECT_ROOT / "logs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# Shared CNN embedding cache used by Step 09.
# Step 08 writes both the canonical step08 name and the step09 compatibility alias.
STEP08_EMBEDDING_CACHE_NPZ = METADATA_DIR / "step08_cnn_embeddings.npz"
STEP08_EMBEDDING_CACHE_INDEX_CSV = METADATA_DIR / "step08_cnn_embeddings_index.csv"
STEP09_COMPAT_EMBEDDING_CACHE_NPZ = METADATA_DIR / "step09_cnn_embedding_cache.npz"
STEP09_COMPAT_EMBEDDING_CACHE_INDEX_CSV = METADATA_DIR / "step09_cnn_embedding_cache_index.csv"

TARGET_COLUMNS = ["target_er", "target_pr", "target_her2"]
LABEL_BINS     = {"target_er": "ER_bin", "target_pr": "PR_bin", "target_her2": "HER2_bin"}
TARGET_DISPLAY = {"target_er": "ER",     "target_pr": "PR",     "target_her2": "HER2"}
GLOBAL_SEED    = 42


# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Step08BConfig:
    project_root:      str   = str(PROJECT_ROOT)
    step_name:         str   = STEP_NAME
    target_columns:    List[str] = field(default_factory=lambda: TARGET_COLUMNS.copy())
    global_seed:       int   = GLOBAL_SEED
    # CNN settings
    backbone:          str   = "efficientnet_b0"   # or "resnet18"
    pretrained:        bool  = True
    freeze_backbone_epochs: int = 10               # freeze CNN for first N epochs
    image_size:        int   = 64                  # resize each 2D slice to this
    max_lesions_per_patient: int = 8               # cap for GPU memory
    # MIL head
    embed_dim:         int   = 128
    attention_dim:     int   = 64
    dropout:           float = 0.4
    # Training
    n_epochs:          int   = 60                 # small cohort: avoid long overfitting runs
    lr_cnn:            float = 5e-5               # conservative CNN fine-tuning
    lr_head:           float = 2e-4               # attention head + classifier
    weight_decay:      float = 5e-4
    patience:          int   = 18                 # early stopping for small validation set
    batch_size:        int   = 8                  # patients per batch (GPU-friendly)
    min_valid_specificity: float = 0.20           # prevent degenerate all-positive thresholds
    # Augmentation
    aug_flip:          bool  = True
    aug_rotate:        bool  = True
    # Loss / sampler / AMP
    use_focal_loss:    bool  = True
    focal_gamma:       float = 2.0
    label_smoothing:   float = 0.05
    use_weighted_sampler: bool = True
    use_amp:           bool  = True
    strong_augmentation: bool = True
    grad_clip_norm:    float = 1.0
    # Evaluation
    threshold_grid: List[float] = field(default_factory=lambda: [0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65])
    n_bootstrap:       int   = 1000
    save_plots:        bool  = True
    save_package:      bool  = True
    # Shared CNN embedding cache for Step 09
    save_embedding_cache: bool = True
    embedding_cache_npz: str = str(STEP08_EMBEDDING_CACHE_NPZ)
    embedding_cache_index_csv: str = str(STEP08_EMBEDDING_CACHE_INDEX_CSV)
    step09_compat_embedding_cache_npz: str = str(STEP09_COMPAT_EMBEDDING_CACHE_NPZ)
    step09_compat_embedding_cache_index_csv: str = str(STEP09_COMPAT_EMBEDDING_CACHE_INDEX_CSV)


# ════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


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
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def normalize_label_pm(x) -> Optional[int]:
    if pd.isna(x): return None
    s = str(x).strip()
    if s in ("+", "1", "1.0"): return 1
    if s in ("-", "0", "0.0"): return 0
    return None


# ════════════════════════════════════════════════════════════════════════════
# NIfTI IMAGE LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_volume(path: str) -> Tuple[np.ndarray, Tuple[float, ...]]:
    """Load NIfTI volume → (H, W, D) float32 array + voxel spacing."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Volume not found: {path}")
    if NIFTI_BACKEND == "nibabel":
        img    = nib.load(str(p))
        arr    = np.asarray(img.get_fdata(), dtype=np.float32)
        zooms  = tuple(float(z) for z in img.header.get_zooms()[:3])
    else:
        img    = sitk.ReadImage(str(p))
        arr    = sitk.GetArrayFromImage(img).astype(np.float32).transpose(2, 1, 0)
        sp     = img.GetSpacing()
        zooms  = (float(sp[0]), float(sp[1]), float(sp[2]))
    return arr, zooms


def normalize_volume(vol: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Z-score normalize within the masked lesion region, then clip."""
    roi_vals = vol[mask > 0]
    if roi_vals.size == 0:
        return vol
    mu  = float(np.mean(roi_vals))
    std = float(np.std(roi_vals)) + 1e-8
    vol = (vol - mu) / std
    return np.clip(vol, -3.0, 3.0).astype(np.float32)


def extract_2p5d_slices(image_path: str, mask_path: str, img_size: int = 64) -> Optional[np.ndarray]:
    """
    Extract 2.5D representation: best slice from each of 3 planes (axial, coronal, sagittal).
    Returns (3, img_size, img_size) float32 array — ready for 3-channel CNN.
    """
    try:
        vol,  _ = load_volume(image_path)
        mask, _ = load_volume(mask_path)
        mask_bin = (mask > 0.5).astype(np.uint8)

        if mask_bin.sum() == 0:
            return None

        # Normalize intensity using lesion ROI
        vol = normalize_volume(vol, mask_bin)

        # Find bounding box centroid
        coords  = np.argwhere(mask_bin)
        centroid = coords.mean(axis=0).astype(int)  # (x, y, z)
        cx, cy, cz = int(centroid[0]), int(centroid[1]), int(centroid[2])

        planes = []
        for axis in range(3):
            if axis == 0:
                sl = vol[cx, :, :]
            elif axis == 1:
                sl = vol[:, cy, :]
            else:
                sl = vol[:, :, cz]

            # Resize to target size
            sl_resized = _resize_slice(sl, img_size)
            planes.append(sl_resized)

        # Stack as 3-channel image (C, H, W)
        result = np.stack(planes, axis=0).astype(np.float32)  # (3, img_size, img_size)
        # Normalize to [0, 1] for pretrained model compatibility
        for c in range(3):
            mn, mx = result[c].min(), result[c].max()
            if mx > mn:
                result[c] = (result[c] - mn) / (mx - mn)
        return result

    except Exception:
        return None


def _resize_slice(sl: np.ndarray, size: int) -> np.ndarray:
    """Simple nearest-neighbor resize of a 2D slice."""
    h, w = sl.shape
    if h == 0 or w == 0:
        return np.zeros((size, size), dtype=np.float32)
    rows = np.linspace(0, h - 1, size).astype(int)
    cols = np.linspace(0, w - 1, size).astype(int)
    return sl[np.ix_(rows, cols)]


# ════════════════════════════════════════════════════════════════════════════
# DATASET
# ════════════════════════════════════════════════════════════════════════════

class PatientBagDataset(Dataset):
    """
    Each patient is a bag of 2.5D CNN feature embeddings.
    Pre-extracts all slices at init time to avoid repeated disk I/O.
    """

    def __init__(
        self,
        patient_records: List[Dict],
        cfg: Step08BConfig,
        augment: bool = False,
    ):
        self.records = patient_records
        self.cfg     = cfg
        self.augment = augment

        # Augmentation transforms (applied per-slice)
        aug_list = [T.ToTensor()]
        if augment:
            if getattr(cfg, "strong_augmentation", True):
                aug_list.extend([
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.25),
                    T.RandomRotation(degrees=20),
                    T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.90, 1.10)),
                    T.ColorJitter(brightness=0.10, contrast=0.10),
                ])
            else:
                if getattr(cfg, "aug_flip", True):
                    aug_list.append(T.RandomHorizontalFlip(p=0.5))
                if getattr(cfg, "aug_rotate", True):
                    aug_list.append(T.RandomRotation(degrees=15))
        aug_list.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]))
        self.transform_aug   = T.Compose(aug_list)
        self.transform_noaug = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rec     = self.records[idx]
        slices  = rec["slices"]   # (N_lesions, 3, H, W) np.float32
        label   = rec["label"]
        n       = len(slices)

        # Cap per GPU memory. Random subset during training (augmentation/variety),
        # but a deterministic head slice for valid/test so locked predictions are
        # reproducible across eval passes (matches step 09's deterministic [:max] cap).
        if n > self.cfg.max_lesions_per_patient:
            if self.augment:
                chosen = np.random.choice(n, self.cfg.max_lesions_per_patient, replace=False)
            else:
                chosen = np.arange(self.cfg.max_lesions_per_patient)
            slices = slices[chosen]

        transform = self.transform_aug if self.augment else self.transform_noaug
        tensors   = []
        for sl in slices:
            # sl: (3, H, W) float32 → PIL-compatible (H, W, 3) uint8
            sl_hwc = np.transpose(sl, (1, 2, 0))           # (H, W, 3)
            sl_u8  = (sl_hwc * 255).clip(0, 255).astype(np.uint8)
            from PIL import Image
            pil_img = Image.fromarray(sl_u8)
            tensors.append(transform(pil_img))              # (3, H, W)

        bag_tensor = torch.stack(tensors)                   # (N, 3, H, W)
        mask       = torch.ones(len(tensors), dtype=torch.bool)
        return bag_tensor, mask, torch.tensor(label, dtype=torch.float32)


def collate_patient_bags(batch):
    bags, masks, labels = zip(*batch)
    max_n = max(b.shape[0] for b in bags)
    c, h, w = bags[0].shape[1:]
    padded  = torch.zeros(len(bags), max_n, c, h, w)
    pad_mask = torch.zeros(len(bags), max_n, dtype=torch.bool)
    for i, (b, m) in enumerate(zip(bags, masks)):
        n = b.shape[0]
        padded[i, :n]   = b
        pad_mask[i, :n] = m
    return padded, pad_mask, torch.stack(labels)


# ════════════════════════════════════════════════════════════════════════════
# MODEL: 2.5D CNN + Attention Pooling
# ════════════════════════════════════════════════════════════════════════════

class CNN25D_MIL(nn.Module):
    """
    Per-lesion: pretrained EfficientNet-B0 / ResNet-18 → embedding
    Per-patient: Gated attention pooling → binary classifier
    """

    def __init__(self, cfg: Step08BConfig, cnn_out_dim: int):
        super().__init__()
        # Attention pooling head
        self.attention_V  = nn.Linear(cnn_out_dim, cfg.attention_dim)
        self.attention_U  = nn.Linear(cnn_out_dim, cfg.attention_dim)
        self.attention_w  = nn.Linear(cfg.attention_dim, 1)
        self.classifier   = nn.Sequential(
            nn.LayerNorm(cnn_out_dim),
            nn.Linear(cnn_out_dim, cfg.embed_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.embed_dim, 1),
        )

    def pool_and_classify(self, h: torch.Tensor, mask: torch.Tensor):
        """h: (B, N, D), mask: (B, N)"""
        A = self.attention_w(torch.tanh(self.attention_V(h)) * torch.sigmoid(self.attention_U(h))).squeeze(-1)
        A = A.masked_fill(~mask, float("-inf"))
        A = F.softmax(A, dim=1)
        A = torch.nan_to_num(A, nan=0.0)
        z = (A.unsqueeze(-1) * h).sum(dim=1)
        return self.classifier(z).squeeze(-1), A


def build_cnn_backbone(cfg: Step08BConfig) -> Tuple[nn.Module, int]:
    """Return (backbone, output_feature_dim)."""
    if cfg.backbone == "efficientnet_b0":
        weights = tv_models.EfficientNet_B0_Weights.DEFAULT if cfg.pretrained else None
        model   = tv_models.efficientnet_b0(weights=weights)
        out_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()
        return model, out_dim
    elif cfg.backbone == "resnet18":
        weights = tv_models.ResNet18_Weights.DEFAULT if cfg.pretrained else None
        model   = tv_models.resnet18(weights=weights)
        out_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, out_dim
    else:
        raise ValueError(f"Unknown backbone: {cfg.backbone}")


def cnn_cache_signature(cfg: Step08BConfig, lesion_df: pd.DataFrame) -> str:
    """Signature shared with Step 09 to prevent stale embedding-cache reuse."""
    return f"{cfg.backbone}|pre={cfg.pretrained}|size={cfg.image_size}|rows={len(lesion_df)}"


def _write_embedding_cache(cache: Dict[int, np.ndarray], npz_path: Path, index_path: Path, signature: str) -> None:
    ensure_dir(npz_path.parent)
    arrays: Dict[str, np.ndarray] = {}
    rows: List[Dict[str, Any]] = []
    for n, (row_index, emb) in enumerate(sorted(cache.items())):
        key = f"emb_{n:06d}"
        arrays[key] = np.asarray(emb, dtype=np.float32)
        rows.append({"row_index": int(row_index), "cache_key": key, "signature": signature})
    np.savez_compressed(npz_path, **arrays)
    pd.DataFrame(rows).to_csv(index_path, index=False, encoding="utf-8-sig")


def build_or_refresh_step08_embedding_cache(
    lesion_df: pd.DataFrame,
    cfg: Step08BConfig,
    device: torch.device,
    logger: logging.Logger,
) -> None:
    """Build a target-independent CNN embedding cache for Step 09."""
    if not getattr(cfg, "save_embedding_cache", True):
        return
    npz_path = Path(getattr(cfg, "embedding_cache_npz", STEP08_EMBEDDING_CACHE_NPZ))
    idx_path = Path(getattr(cfg, "embedding_cache_index_csv", STEP08_EMBEDDING_CACHE_INDEX_CSV))
    signature = cnn_cache_signature(cfg, lesion_df)
    if npz_path.exists() and idx_path.exists():
        try:
            idx_df = pd.read_csv(idx_path)
            if "signature" in idx_df.columns and idx_df["signature"].dropna().nunique() == 1:
                if str(idx_df["signature"].dropna().iloc[0]) == signature:
                    logger.info("Shared CNN embedding cache already current: %s (%d embeddings)", npz_path, len(idx_df))
                    compat_npz = Path(getattr(cfg, "step09_compat_embedding_cache_npz", STEP09_COMPAT_EMBEDDING_CACHE_NPZ))
                    compat_idx = Path(getattr(cfg, "step09_compat_embedding_cache_index_csv", STEP09_COMPAT_EMBEDDING_CACHE_INDEX_CSV))
                    if not compat_npz.exists() or not compat_idx.exists():
                        import shutil
                        shutil.copy2(npz_path, compat_npz)
                        shutil.copy2(idx_path, compat_idx)
                        logger.info("Created Step 09 compatibility cache aliases.")
                    return
        except Exception as e:
            logger.warning("Could not validate existing Step 08 embedding cache; rebuilding. Reason: %s", e)
    img_col = next((c for c in ["image_abs_path", "image_path"] if c in lesion_df.columns), None)
    mask_col = next((c for c in ["mask_abs_path", "mask_path"] if c in lesion_df.columns), None)
    if img_col is None or mask_col is None:
        logger.warning("Cannot build Step 08 embedding cache: image/mask columns are missing.")
        return
    logger.info("Building shared Step 08 CNN embedding cache for Step 09...")
    backbone, _ = build_cnn_backbone(cfg)
    backbone = backbone.to(device).eval()
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    cache: Dict[int, np.ndarray] = {}
    n_ok = 0
    n_fail = 0
    from PIL import Image
    with torch.no_grad():
        for idx, row in lesion_df.iterrows():
            ip = str(row[img_col]) if pd.notna(row[img_col]) else ""
            mp = str(row[mask_col]) if pd.notna(row[mask_col]) else ""
            if not ip or not mp:
                n_fail += 1
                continue
            sl = extract_2p5d_slices(ip, mp, img_size=cfg.image_size)
            if sl is None:
                n_fail += 1
                continue
            sl_hwc = np.transpose(sl, (1, 2, 0))
            sl_u8 = (sl_hwc * 255).clip(0, 255).astype(np.uint8)
            tensor = transform(Image.fromarray(sl_u8)).unsqueeze(0).to(device)
            emb = backbone(tensor).squeeze(0).cpu().numpy()
            cache[int(idx)] = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            n_ok += 1
    if cache:
        _write_embedding_cache(cache, npz_path, idx_path, signature)
        compat_npz = Path(getattr(cfg, "step09_compat_embedding_cache_npz", STEP09_COMPAT_EMBEDDING_CACHE_NPZ))
        compat_idx = Path(getattr(cfg, "step09_compat_embedding_cache_index_csv", STEP09_COMPAT_EMBEDDING_CACHE_INDEX_CSV))
        _write_embedding_cache(cache, compat_npz, compat_idx, signature)
        logger.info("Saved shared CNN embedding cache: %d ok, %d failed | %s", n_ok, n_fail, npz_path)
    else:
        logger.warning("No embeddings were cached for Step 09; all %d rows failed extraction.", n_fail)


class FullModel(nn.Module):
    def __init__(self, backbone: nn.Module, mil_head: CNN25D_MIL):
        super().__init__()
        self.backbone = backbone
        self.mil_head = mil_head

    def forward(self, bags: torch.Tensor, mask: torch.Tensor):
        """
        bags : (B, N, 3, H, W)
        mask : (B, N)
        """
        B, N, C, H, W = bags.shape
        # Extract per-lesion features
        flat  = bags.view(B * N, C, H, W)
        feats = self.backbone(flat)                   # (B*N, D)
        feats = feats.view(B, N, -1)                  # (B, N, D)
        logit, attn = self.mil_head.pool_and_classify(feats, mask)
        return logit, attn


# ════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ════════════════════════════════════════════════════════════════════════════

def prepare_patient_records(
    lesion_df: pd.DataFrame,
    target_col: str,
    splits_df: pd.DataFrame,
    cfg: Step08BConfig,
    logger: logging.Logger,
) -> List[Dict]:
    """
    For each patient: extract 2.5D slices for all lesions, store as pre-loaded numpy arrays.
    Returns list of patient dicts ready for Dataset.
    """
    label_src  = LABEL_BINS.get(target_col, "")
    df         = lesion_df.copy()
    df         = df.merge(splits_df[["patient_base", "split"]], on="patient_base", how="left")
    df["_lbl"] = df[label_src].apply(normalize_label_pm) if label_src in df.columns else None
    df         = df[df["_lbl"].notna()].copy()

    # Resolve image/mask path columns
    img_col  = next((c for c in ["image_abs_path", "image_path"] if c in df.columns), None)
    mask_col = next((c for c in ["mask_abs_path",  "mask_path"]  if c in df.columns), None)
    if img_col is None or mask_col is None:
        raise ValueError("No image/mask path columns found in lesion CSV.")

    records  = []
    n_ok     = 0
    n_fail   = 0
    patients = df.groupby("patient_base")

    for patient_id, grp in patients:
        label_vals = grp["_lbl"].unique()
        if len(label_vals) > 1:
            continue  # conflicting labels
        label = int(label_vals[0])
        split_vals = grp["split"].dropna().unique() if "split" in grp.columns else ["train"]
        if len(split_vals) != 1:
            continue  # skip patients with inconsistent/missing split assignment (matches steps 07/09)
        split = str(split_vals[0])

        patient_slices = []
        for _, row in grp.iterrows():
            img_path  = str(row[img_col])  if pd.notna(row[img_col])  else ""
            msk_path  = str(row[mask_col]) if pd.notna(row[mask_col]) else ""
            if not img_path or not msk_path:
                continue
            sl = extract_2p5d_slices(img_path, msk_path, img_size=cfg.image_size)
            if sl is not None:
                patient_slices.append(sl)
                n_ok += 1
            else:
                n_fail += 1

        if not patient_slices:
            continue

        slices_arr = np.stack(patient_slices, axis=0)  # (N_lesions, 3, H, W)
        records.append({
            "patient_id": patient_id,
            "label":      label,
            "split":      split,
            "slices":     slices_arr,
            "n_lesions":  len(patient_slices),
        })

    logger.info("  Slice extraction: %d ok, %d failed | %d patients with valid slices",
                n_ok, n_fail, len(records))
    return records


# ════════════════════════════════════════════════════════════════════════════
# METRICS & PLOTS
# ════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, probs, preds) -> Dict:
    if len(np.unique(y_true)) < 2:
        return {"AUROC": float("nan"), "AUPRC": float("nan"),
                "F1": float("nan"), "Balanced_Acc": float("nan")}
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    sens = float(tp / max(tp + fn, 1))
    spec = float(tn / max(tn + fp, 1))
    is_degenerate = (spec < 0.05) or (sens < 0.05)
    return {
        "AUROC":        float(roc_auc_score(y_true, probs)),
        "AUPRC":        float(average_precision_score(y_true, probs)),
        "F1":           float(f1_score(y_true, preds, zero_division=0)),
        "Balanced_Acc": float(balanced_accuracy_score(y_true, preds)),
        "Sensitivity":  sens,
        "Specificity":  spec,
        "PPV_Precision": float(tp / max(tp + fp, 1)),
        "NPV":           float(tn / max(tn + fn, 1)),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "degenerate_prediction": bool(is_degenerate),
        "MCC": float(matthews_corrcoef(y_true, preds)),
    }

def bootstrap_auroc(y_true, probs, n=1000, seed=42):
    rng  = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        yb, pb = y_true[idx], probs[idx]
        if len(np.unique(yb)) < 2: continue
        aucs.append(roc_auc_score(yb, pb))
    if len(aucs) < 100: return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def choose_threshold(y_true, probs, grid, min_specificity: float = 0.20):
    best_t, best_ba = 0.5, -1.0
    best_t_fb, best_fb = 0.5, -1.0
    for t in grid:
        preds = (probs >= t).astype(int)
        ba = balanced_accuracy_score(y_true, preds)
        tn = int(((y_true == 0) & (preds == 0)).sum())
        fp = int(((y_true == 0) & (preds == 1)).sum())
        spec = tn / max(tn + fp, 1)
        if ba > best_fb:
            best_fb, best_t_fb = ba, t
        if ba > best_ba and spec >= min_specificity:
            best_ba, best_t = ba, t
    if best_ba < 0:
        return best_t_fb
    return best_t

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


def plot_training_curves(history, path, title):
    """Training curves showing loss, real AUROC, and composite early-stopping score."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    epochs = [h["epoch"] for h in history]

    # Panel 1: Loss
    axes[0].plot(epochs, [h["train_loss"] for h in history], color="#2E75B6", label="train")
    axes[0].plot(epochs, [h["valid_loss"] for h in history], color="#C00000", linestyle="--", label="valid")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Panel 2: Real AUROC (what matters clinically)
    axes[1].plot(epochs, [h["valid_auroc"] for h in history], color="#375623", lw=2, label="Real AUROC")
    axes[1].axhline(0.5, linestyle="--", color="gray", alpha=0.5, label="Random")
    axes[1].set_title("Validation AUROC (real)"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1); axes[1].legend(); axes[1].grid(alpha=0.3)

    # Panel 3: Composite score (used for early stopping — 0.7*AUROC + 0.3*BA - spec_penalty)
    if "valid_composite" in history[0]:
        axes[2].plot(epochs, [h["valid_composite"] for h in history], color="#7B2C9E", lw=2, label="Composite")
        axes[2].axhline(0.5, linestyle="--", color="gray", alpha=0.5)
        axes[2].set_title("Composite Score (early stopping)"); axes[2].set_xlabel("Epoch")
        axes[2].set_ylim(0, 1); axes[2].legend(); axes[2].grid(alpha=0.3)
        axes[2].annotate("= 0.7×AUROC + 0.3×BA − spec_penalty",
                         xy=(0.02, 0.04), xycoords="axes fraction", fontsize=7, color="#7B2C9E")
    else:
        axes[2].axis("off")

    plt.suptitle(title, fontsize=12); plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()


def plot_summary(all_rows, path):
    df   = pd.DataFrame(all_rows)
    if df.empty: return
    lbls = df["target_name"] + "\n" + df["backbone"]
    vals = df["test_AUROC"].tolist()
    fig, ax = plt.subplots(figsize=(max(8, len(vals)*1.8), 5))
    bars = ax.bar(range(len(vals)), vals, color="#1F4E79")
    ax.axhline(0.5, linestyle="--", color="gray", lw=1)
    ax.set_xticks(range(len(lbls))); ax.set_xticklabels(lbls, rotation=25, ha="right")
    ax.set_ylabel("Test AUROC"); ax.set_ylim(0, 1.05)
    ax.set_title("Step 08B — 2.5D CNN Transfer Learning Results")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()




class FocalBCEWithLogitsLoss(nn.Module):
    """Binary focal loss with optional pos_weight and light label smoothing."""

    def __init__(self, pos_weight: Optional[torch.Tensor] = None, gamma: float = 1.5, label_smoothing: float = 0.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        if self.label_smoothing > 0:
            eps = float(self.label_smoothing)
            targets = targets * (1.0 - eps) + 0.5 * eps
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        focal = (1.0 - p_t).clamp(min=1e-6).pow(self.gamma)
        return (focal * bce).mean()


def make_weighted_sampler(records: List[Dict]) -> Optional[WeightedRandomSampler]:
    """Balanced patient-level sampler for small imbalanced train splits."""
    labels = np.array([int(r["label"]) for r in records], dtype=int)
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return None
    counts = np.bincount(labels, minlength=2).astype(float)
    counts[counts == 0] = 1.0
    weights = 1.0 / counts
    sample_weights = torch.as_tensor([weights[y] for y in labels], dtype=torch.double)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# ════════════════════════════════════════════════════════════════════════════
# TRAINING ENGINE
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_cnn(model, loader, device, criterion: Optional[nn.Module] = None):
    model.eval()
    all_probs, all_labels, total_loss, n_samples = [], [], 0.0, 0
    for bags, mask, labels in loader:
        bags, mask = bags.to(device), mask.to(device)
        labels_dev = labels.to(device)
        logits, _ = model(bags, mask)
        if criterion is not None:
            total_loss += float(criterion(logits, labels_dev).item()) * len(labels)
            n_samples += len(labels)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    avg_loss = total_loss / max(n_samples, 1) if criterion is not None else float("nan")
    return np.array(all_probs), np.array(all_labels, dtype=int), avg_loss


def train_cnn_mil(records, target_col, cfg, device, logger):
    tname = TARGET_DISPLAY.get(target_col, target_col)

    train_recs = [r for r in records if r["split"] == "train"]
    valid_recs = [r for r in records if r["split"] == "valid"]
    test_recs  = [r for r in records if r["split"] == "test"]

    if not train_recs or not valid_recs or not test_recs:
        logger.warning("  Insufficient patient records for %s — skipping.", tname)
        return None

    logger.info("  CNN-MIL train=%d valid=%d test=%d", len(train_recs), len(valid_recs), len(test_recs))

    try:
        from PIL import Image  # check PIL is available
    except ImportError:
        raise RuntimeError("Install Pillow: pip install Pillow")

    train_ds = PatientBagDataset(train_recs, cfg, augment=True)
    valid_ds = PatientBagDataset(valid_recs, cfg, augment=False)
    test_ds  = PatientBagDataset(test_recs,  cfg, augment=False)

    kw = dict(collate_fn=collate_patient_bags, num_workers=0, pin_memory=(device.type=="cuda"))
    if getattr(cfg, "use_weighted_sampler", True):
        _sampler = make_weighted_sampler(train_recs)
    else:
        _sampler = None
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=(_sampler is None), sampler=_sampler, **kw)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, **kw)

    backbone, cnn_out_dim = build_cnn_backbone(cfg)
    mil_head              = CNN25D_MIL(cfg, cnn_out_dim)
    model                 = FullModel(backbone, mil_head).to(device)

    n_pos = sum(r["label"] for r in train_recs)
    n_neg = len(train_recs) - n_pos
    pos_w = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32, device=device)
    if getattr(cfg, "use_focal_loss", True):
        criterion = FocalBCEWithLogitsLoss(
            pos_weight=pos_w,
            gamma=getattr(cfg, "focal_gamma", 2.0),
            label_smoothing=getattr(cfg, "label_smoothing", 0.05),
        )
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    # Separate LR for CNN backbone vs MIL head
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(),  "lr": cfg.lr_cnn},
        {"params": model.mil_head.parameters(),  "lr": cfg.lr_head},
    ], weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_epochs)

    best_composite_score  = -1.0   # tracks composite = 0.7*AUROC + 0.3*BA - spec_penalty
    best_valid_auroc_real = -1.0   # tracks the real AUROC at the best checkpoint
    best_state            = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    patience_count        = 0
    history               = []
    _use_amp = getattr(cfg, "use_amp", True) and device.type == "cuda"
    scaler_amp = make_grad_scaler(enabled=_use_amp)

    for epoch in range(1, cfg.n_epochs + 1):
        # Freeze backbone for first N epochs
        for p in model.backbone.parameters():
            p.requires_grad = (epoch > cfg.freeze_backbone_epochs)

        model.train()
        train_loss = 0.0
        for bags, mask, labels in train_loader:
            bags, mask, labels = bags.to(device), mask.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast_context(enabled=_use_amp):
                logits, _ = model(bags, mask)
                loss = criterion(logits, labels)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=getattr(cfg, "grad_clip_norm", 1.0))
            scaler_amp.step(optimizer)
            scaler_amp.update()
            train_loss += loss.item() * len(labels)
        scheduler.step()

        train_loss /= max(len(train_loader.dataset), 1)
        v_probs, v_labels, v_loss = evaluate_cnn(model, valid_loader, device, criterion=criterion)
        v_auroc = roc_auc_score(v_labels, v_probs) if len(np.unique(v_labels)) > 1 else 0.5

        v_preds_05 = (v_probs >= 0.5).astype(int)
        v_ba_05 = balanced_accuracy_score(v_labels, v_preds_05)
        v_tn_05 = int(((v_labels == 0) & (v_preds_05 == 0)).sum())
        v_fp_05 = int(((v_labels == 0) & (v_preds_05 == 1)).sum())
        v_spec_05 = v_tn_05 / max(v_tn_05 + v_fp_05, 1)
        spec_penalty = 0.1 if v_spec_05 < 0.1 else 0.0
        composite = 0.7 * v_auroc + 0.3 * v_ba_05 - spec_penalty

        history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                        "valid_loss": round(v_loss, 4),
                        "valid_auroc": round(v_auroc, 4),
                        "valid_composite": round(composite, 4)})

        if composite > best_composite_score:
            best_composite_score  = composite
            best_valid_auroc_real = v_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if epoch % 10 == 0:
            logger.info("  Epoch %3d | train_loss=%.4f | valid_AUROC=%.4f | composite=%.4f (best=%.4f)",
                        epoch, train_loss, v_auroc, composite, best_composite_score)
        if patience_count >= cfg.patience:
            logger.info("  Early stop at epoch %d", epoch); break

    model.load_state_dict(best_state)

    # Validation-only threshold selection
    v_probs, v_labels, v_loss_final = evaluate_cnn(model, valid_loader, device, criterion=criterion)
    best_thr   = choose_threshold(v_labels, v_probs, cfg.threshold_grid,
                                 min_specificity=cfg.min_valid_specificity)
    valid_mets = compute_metrics(v_labels, v_probs, (v_probs >= best_thr).astype(int))
    valid_mets["Loss"] = float(v_loss_final)

    # Locked test
    t_probs, t_labels, t_loss_final = evaluate_cnn(model, test_loader, device, criterion=criterion)
    t_preds   = (t_probs >= best_thr).astype(int)
    test_mets = compute_metrics(t_labels, t_probs, t_preds)
    test_mets["Loss"] = float(t_loss_final)
    ci_lo, ci_hi = bootstrap_auroc(t_labels, t_probs, n=cfg.n_bootstrap)
    test_mets["AUROC_CI95_lower"] = ci_lo
    test_mets["AUROC_CI95_upper"] = ci_hi
    test_mets["generalization_gap"] = float(valid_mets["AUROC"] - test_mets["AUROC"])

    return {
        "model":           model,
        "history":         history,
        "best_threshold":  best_thr,
        "valid_metrics":   valid_mets,
        "test_metrics":    test_mets,
        "test_probs":      t_probs,
        "test_labels":     t_labels,
        "valid_probs":     v_probs,
        "valid_labels":    v_labels,
        "best_composite_score":  best_composite_score,
        "best_valid_auroc":       best_valid_auroc_real,  # real AUROC at best checkpoint (not composite)
        "backbone":        cfg.backbone,
        "n_train":         len(train_recs),
        "n_valid":         len(valid_recs),
        "n_test":          len(test_recs),
    }


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _ap = argparse.ArgumentParser(add_help=False)
    _ap.add_argument("--force",    action="store_true")
    _ap.add_argument("--no-cache", action="store_true")
    _flags, _ = _ap.parse_known_args()

    cfg    = Step08BConfig()
    set_seed(cfg.global_seed)

    for p in [REPORTS_DIR, FIGURES_DIR, TABLES_DIR, METADATA_DIR, LOGS_DIR, CHECKPOINTS_DIR]:
        ensure_dir(p)

    log_file = LOGS_DIR / f"{STEP_NAME}.log"
    logger   = setup_logger(log_file)
    device   = get_device()
    logger.info("Device: %s | Backbone: %s | Image size: %d",
                device, cfg.backbone, cfg.image_size)

    # ── Cache check ───────────────────────────────────────────────────────────
    _cfg_hash = hashlib.md5(json.dumps({
        "backbone": cfg.backbone, "n_epochs": cfg.n_epochs,
        "seed": cfg.global_seed, "image_size": cfg.image_size,
    }, sort_keys=True).encode()).hexdigest()
    _manifest = PROJECT_ROOT / "cache" / "cnn_embeddings" / "step08_cache_manifest.json"
    _inputs   = [INPUT_LESION_CSV, INPUT_SPLITS_CSV]
    _outputs  = [TABLES_DIR / "step08_cnn_model_results.csv", METADATA_DIR / "step08_cnn_summary.json"]
    if not _flags.force and not _flags.no_cache and _is_step_cached(_manifest, _inputs, _cfg_hash, _outputs):
        logger.info("Cache valid — Step 08 outputs unchanged. Skipping (use --force to re-run).")
        return

    if not INPUT_LESION_CSV.exists():
        raise FileNotFoundError(f"Missing: {INPUT_LESION_CSV}")

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

    # Derive binary labels if needed
    for src, bc in [("ER","ER_bin"),("PR","PR_bin"),("HER2","HER2_bin")]:
        if bc not in lesion_df.columns and src in lesion_df.columns:
            lesion_df[bc] = lesion_df[src].apply(normalize_label_pm)

    # Build a target-independent CNN embedding cache once for Step 09.
    # This does not affect Step 08 training outputs; it only avoids repeated NIfTI I/O later.
    try:
        build_or_refresh_step08_embedding_cache(lesion_df, cfg, device, logger)
    except Exception as e:
        logger.warning("Step 08 embedding cache build failed but Step 08 training will continue: %s", e)

    all_rows = []

    for target_col in cfg.target_columns:
        tname = TARGET_DISPLAY.get(target_col, target_col)
        logger.info("=" * 60)
        logger.info("Target: %s", tname)

        records = prepare_patient_records(lesion_df, target_col, splits_df, cfg, logger)
        if len(records) < 10:
            logger.warning("  Only %d patients with valid slices — skipping %s.", len(records), tname)
            continue

        set_seed(cfg.global_seed)
        result = train_cnn_mil(records, target_col, cfg, device, logger)
        if result is None:
            continue

        tm = result["test_metrics"]
        degen_flag = "⚠️ DEGENERATE" if tm.get("degenerate_prediction") else "✓ OK"
        logger.info("  [%s | %s] %s | Test AUROC=%.4f (CI: %.3f–%.3f) | AUPRC=%.4f | BA=%.4f | Sens=%.3f | Spec=%.3f",
                    tname, cfg.backbone, degen_flag,
                    tm.get("AUROC", float("nan")),
                    tm.get("AUROC_CI95_lower", float("nan")),
                    tm.get("AUROC_CI95_upper", float("nan")),
                    tm.get("AUPRC", float("nan")),
                    tm.get("Balanced_Acc", float("nan")),
                    tm.get("Sensitivity", float("nan")),
                    tm.get("Specificity", float("nan")))
        if tm.get("degenerate_prediction"):
            logger.warning("  ⚠️  [%s | %s] Degenerate prediction detected (Sens=%.2f, Spec=%.2f).",
                           tname, cfg.backbone,
                           tm.get("Sensitivity", 0.0), tm.get("Specificity", 0.0))

        sfx = f"{target_col}_{cfg.backbone}"
        if cfg.save_plots:
            plot_training_curves(
                result["history"],
                FIGURES_DIR / f"step08_{sfx}_training.png",
                f"{tname} | 2.5D CNN ({cfg.backbone})"
            )
            if len(np.unique(result["test_labels"])) > 1:
                plot_roc_pr(
                    result["test_labels"], result["test_probs"],
                    FIGURES_DIR / f"step08_{sfx}_roc_pr.png",
                    f"{tname} | 2.5D CNN ({cfg.backbone}) — Locked Test"
                )
        pd.DataFrame(result["history"]).to_csv(
            TABLES_DIR / f"step08_{sfx}_history.csv", index=False, encoding="utf-8-sig"
        )

        row = {"target": target_col, "target_name": tname, "backbone": cfg.backbone,
               "n_train": result["n_train"], "n_valid": result["n_valid"],
               "n_test": result["n_test"], "best_threshold": result["best_threshold"],
               "valid_AUROC":            result["best_valid_auroc"],       # real AUROC at best checkpoint
               "valid_composite_score":  result["best_composite_score"],  # composite used for early stopping
               **{f"test_{k}": v for k, v in tm.items()}}
        all_rows.append(row)

    if not all_rows:
        logger.error("No results. Ensure NIfTI files are accessible at image_abs_path/mask_abs_path.")
        return

    master_df = pd.DataFrame(all_rows)
    master_df.to_csv(TABLES_DIR / "step08_cnn_model_results.csv", index=False, encoding="utf-8-sig")

    if cfg.save_plots:
        plot_summary(all_rows, FIGURES_DIR / "step08_cnn_comparison.png")

    save_json({"config": asdict(cfg), "device": str(device),
               "results": all_rows}, METADATA_DIR / "step08_cnn_summary.json")
    _save_step_manifest(_manifest, _inputs, _cfg_hash)

    if cfg.save_package:
        zip_path = REPORTS_DIR / "step08_cnn_publication_package.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(TABLES_DIR.glob("step08_*")):
                zf.write(p, f"tables/{p.name}")
            for p in sorted(FIGURES_DIR.glob("step08_*")):
                zf.write(p, f"figures/{p.name}")

    logger.info("Step 08B complete.")


if __name__ == "__main__":
    main()
