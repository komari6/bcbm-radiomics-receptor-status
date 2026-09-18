from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


# =========================================================
# 1) SETTINGS
# =========================================================

PROJECT_ROOT   = Path("./bcbm_project").resolve()
STEP_NAME      = "step_03_segmentation_harmonization"
INPUT_CANONICAL = PROJECT_ROOT / "data" / "interim" / "tabular_raw_canonical.csv"

BUILD_LESION_ONLY_SUBSET    = True
SAVE_PARQUET_IF_AVAILABLE   = True
FORCE_MASK_FEATURE_EXTRACTION = True   # Set False if NIfTI files are absent to save time
ENABLE_MASK_FEATURE_CACHE     = True
ENABLE_EARLY_TEXT_RESOLUTION  = True
PROGRESS_LOG_EVERY = 100

VERY_SMALL_VOXELS = 10
SMALL_VOXELS      = 50
MEDIUM_VOXELS     = 500
LARGE_VOXELS      = 5000

# Validated manual decisions for ambiguous descriptors
VALIDATED_DESCRIPTOR_DECISIONS = {
    "bs":                    "other_structure",
    "right_motor_strip":     "other_structure",
    "r_hiccpocampus":        "other_structure",
    "right_foramen_of_monro":"other_structure",
    "rlatvent":              "other_structure",
    "rtlatcllum":            "other_structure",
    "r_infpar":              "other_structure",
    "pineal_region":         "lesion",
    # FIX I-19: "an" removed — too broad (2-letter match). Add specific terms instead:
    "aneurysm":              "lesion",
    "angioma":               "lesion",
    "l_frntl":               "lesion",
    "r_frntl":               "lesion",
    "latposttem":            "lesion",
    "lcsup":                 "lesion",
    "left_fp":               "lesion",
    "leftvent":              "other_structure",  # FIX I-39: left ventricle is a brain structure, not a lesion
    "lftsupfro":             "lesion",
    "linflatcbm":            "lesion",
    "llatposfro":            "lesion",
    "ltinfcrblm":            "lesion",
    "ltmedposfr":            "lesion",
    "rtinfcrblm":            "lesion",
    "rtmedcrblm":            "lesion",
    "rtpossupcb":            "lesion",
}

STRICT_STRUCTURE_TERMS = [
    "bst", "brainstem", "brain_stem", "brnstm",
    "optic", "optic_nerve", "opticnerve", "opt_nerve", "opt_nerv", "opt_nrv",
    "opt_system", "op_trct", "opticchiasm", "chiasm",
    "lens", "cochlea",
    "hippocampus", "hiccpocampus",
    "pituitary", "pit_stalk", "pituitary_stalk",
    "corpus_callosum",
    "orbit", "eye", "globe", "globes",
    "medulla", "foramen_of_monro",
]

# FIX: added "tum" and "tumo" — common truncations in this dataset
STRICT_LESION_TERMS = [
    "tumor", "tumour", "tum", "tumo",
    "met", "mets", "metastasis",
    "lesion", "gtv",
]

TARGET_TERMS = ["plan", "tgt", "target", "ptv", "ctv", "gy"]

CAVITY_TERMS = [
    "cavity", "bed", "resection", "resect",
    "postop", "post_op", "postoperative",
    "surgery_bed", "surgical_cavity",
]

LESION_LOCATION_TERMS = [
    "front", "frontal",
    "parie", "pariet", "parietal",
    "occ", "occip", "occipital", "occpital",
    "temp", "temporal", "temporo", "temporoparietal",
    "cereb", "cerb", "cerebell", "cerebellar", "cbll", "cbllm", "cbl", "crblm",
    "thal", "thalam", "thalamic",
    "caud", "caudate",
    "ganglia", "basal",
    "capsule",
    "insula", "insul",
    "paramedian", "paraventric", "paraventricular",
    "sylv", "sylvian",
    "pons", "midbrain", "vermis",
    "tentorial", "subtentorial",
    "perivent", "periventricular",
]

LESION_MODIFIER_TERMS = [
    "mrg", "marg", "margin",
    "deep",
    "anterior", "posterior",
    "medial", "lateral",
    "superior", "inferior",
    "retreat", "parasagittal", "motor_strip",
]

# FIX: prefix anchoring متسق مع Stage 2 و Stage 4+5
RADIOMICS_PREFIXES = [
    "original_", "wavelet_", "log_sigma_", "lbp_",
    "shape_", "firstorder_", "glcm_", "glrlm_",
    "glszm_", "gldm_", "ngtdm_", "diagnostics_",
]


# =========================================================
# 2) LOGGING / STATE
# =========================================================

def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger(STEP_NAME)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    formatter = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


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
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
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


# =========================================================
# 3) HELPERS
# =========================================================

def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def normalize_text(value: Any) -> str:
    s = str(value).strip().lower()
    s = s.replace("\\", "/")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def normalize_label_pm(x: Any) -> Optional[int]:
    if pd.isna(x):
        return None
    x = str(x).strip()
    if x == "+":
        return 1
    if x == "-":
        return 0
    return None


def extract_filename_from_path(path_value: Any) -> Optional[str]:
    if pd.isna(path_value):
        return None
    s = str(path_value).replace("\\", "/").strip()
    return Path(s).name if s else None


def strip_nii_gz(filename: Any) -> str:
    if pd.isna(filename):
        return ""
    s = str(filename).strip()
    s = re.sub(r"\.nii\.gz$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\.nii$",    "", s, flags=re.IGNORECASE)
    return s


def extract_mask_core_from_filename(filename: Any) -> str:
    name = strip_nii_gz(filename)
    if not name:
        return ""
    m = re.search(r"_mask_(.+)$", name, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return name


def canonicalize_mask_core(mask_core: Any) -> str:
    s = str(mask_core).strip().lower()
    s = s.replace("\\", "/")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")

    replacements = {
        "rt": "right", "lt": "left",
        "sup": "superior", "ant": "anterior", "post": "posterior",
        "lat": "lateral", "med": "medial", "inf": "inferior",
        "temp": "temporal", "tempo": "temporal",
        "tum": "tumor", "tumo": "tumor",      # FIX: explicit tumor abbreviations
        "cerb": "cereb",
        "crbllr": "cerebellar", "cerebllr": "cerebellar",
        "cerebellr": "cerebellar", "cbllr": "cerebellar",
        "cbllm": "cerebellum", "cblm": "cerebellum", "crblm": "cerebellum",
        "frntl": "frontal", "fro": "frontal",
        "parie": "parietal",
        "occ": "occipital",
        "hiccpocampus": "hippocampus",
        "brnstm": "brainstem",
        # NOTE: "op" -> "optic" removed to avoid false matches;
        # "optic" itself is already in STRICT_STRUCTURE_TERMS
    }

    parts = s.split("_")
    parts = [replacements.get(p, p) for p in parts]
    s = "_".join(parts)
    s = s.replace("temporaloroparietal", "temporoparietal")
    s = s.replace("temporo_parietal", "temporoparietal")
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def choose_best_segmentation_text(seg_name: Any, matched_mask_file: Any, mask_path: Any) -> Tuple[str, str]:
    if not pd.isna(matched_mask_file) and str(matched_mask_file).strip():
        return extract_mask_core_from_filename(str(matched_mask_file).strip()), "matched_mask_file"
    mask_filename = extract_filename_from_path(mask_path)
    if mask_filename:
        return extract_mask_core_from_filename(mask_filename), "mask_path"
    if not pd.isna(seg_name):
        return str(seg_name).strip(), "Segmentation_Name"
    return "", "none"


def is_radiomics_column(col: str) -> bool:
    col_norm = normalize_text(col) + "_"
    return any(col_norm.startswith(pfx) for pfx in RADIOMICS_PREFIXES)


def build_empty_mask_feature_record() -> Dict[str, Any]:
    return {
        "mask_read_ok": None, "mask_read_error": "",
        "mask_voxel_count": None, "mask_volume_mm3": None, "mask_volume_cc": None,
        "bbox_dim_0": None, "bbox_dim_1": None, "bbox_dim_2": None,
        "bbox_volume": None, "mask_fill_ratio_in_bbox": None,
        "n_components": None, "n_slices_involved": None,
        "elongation_ratio_01": None, "elongation_ratio_02": None, "elongation_ratio_12": None,
    }


# =========================================================
# 4) NIFTI READING
# =========================================================

def load_mask_array(mask_path: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    if not mask_path or str(mask_path).strip() == "":
        raise ValueError("Empty mask path")
    path = Path(mask_path)
    if not path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    try:
        import nibabel as nib
        img = nib.load(str(path))
        arr = np.asarray(img.get_fdata())
        zooms = img.header.get_zooms()[:3]
        return arr > 0, tuple(float(z) for z in zooms)
    except Exception:
        pass

    try:
        import SimpleITK as sitk
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img)
        sp  = img.GetSpacing()
        return np.asarray(arr) > 0, (float(sp[2]), float(sp[1]), float(sp[0]))
    except Exception as e:
        raise ImportError(f"Cannot read NIfTI (install nibabel or SimpleITK): {e}")


# =========================================================
# 5) MASK FEATURES
# =========================================================

def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return coords.min(axis=0), coords.max(axis=0)


def compute_mask_features(mask: np.ndarray, spacing: Tuple[float, float, float]) -> Dict[str, Any]:
    voxel_count = int(mask.sum())
    if voxel_count == 0:
        return {
            "mask_voxel_count": 0, "mask_volume_mm3": 0.0, "mask_volume_cc": 0.0,
            "bbox_dim_0": 0, "bbox_dim_1": 0, "bbox_dim_2": 0, "bbox_volume": 0,
            "mask_fill_ratio_in_bbox": 0.0, "n_components": None,
            "n_slices_involved": 0,
            "elongation_ratio_01": None, "elongation_ratio_02": None, "elongation_ratio_12": None,
        }

    bbox = bbox_from_mask(mask)
    if bbox is None:
        return {
            "mask_voxel_count": voxel_count, "mask_volume_mm3": 0.0, "mask_volume_cc": 0.0,
            "bbox_dim_0": 0, "bbox_dim_1": 0, "bbox_dim_2": 0, "bbox_volume": 0,
            "mask_fill_ratio_in_bbox": 0.0, "n_components": None, "n_slices_involved": 0,
            "elongation_ratio_01": None, "elongation_ratio_02": None, "elongation_ratio_12": None,
        }

    coords     = np.argwhere(mask)
    mins, maxs = bbox
    dims       = (maxs - mins + 1).astype(int)

    bbox_volume     = int(np.prod(dims))
    fill_ratio      = float(voxel_count / bbox_volume) if bbox_volume > 0 else 0.0
    voxel_volume    = float(spacing[0] * spacing[1] * spacing[2])
    mask_volume_mm3 = float(voxel_count * voxel_volume)
    n_slices        = int(np.unique(coords[:, 0]).shape[0])

    n_components = None
    try:
        from scipy import ndimage
        _, n_components = ndimage.label(mask.astype(np.uint8))
        n_components = int(n_components)
    except Exception:
        pass

    def ratio(a: int, b: int) -> Optional[float]:
        if b == 0:
            return None
        return float(max(a, b) / max(1, min(a, b)))

    return {
        "mask_voxel_count":         voxel_count,
        "mask_volume_mm3":          round(mask_volume_mm3, 6),
        "mask_volume_cc":           round(mask_volume_mm3 / 1000.0, 6),
        "bbox_dim_0":               int(dims[0]),
        "bbox_dim_1":               int(dims[1]),
        "bbox_dim_2":               int(dims[2]),
        "bbox_volume":              bbox_volume,
        "mask_fill_ratio_in_bbox":  round(fill_ratio, 6),
        "n_components":             n_components,
        "n_slices_involved":        n_slices,
        "elongation_ratio_01":      round(ratio(int(dims[0]), int(dims[1])), 6) if ratio(int(dims[0]), int(dims[1])) is not None else None,
        "elongation_ratio_02":      round(ratio(int(dims[0]), int(dims[2])), 6) if ratio(int(dims[0]), int(dims[2])) is not None else None,
        "elongation_ratio_12":      round(ratio(int(dims[1]), int(dims[2])), 6) if ratio(int(dims[1]), int(dims[2])) is not None else None,
    }


# =========================================================
# 6) DECISION ENGINE
# =========================================================

def contains_any(s: str, terms: list) -> bool:
    return any(t in s for t in terms)


def text_rule_classification(s: str) -> Tuple[Optional[str], str, str]:
    if not s:
        return None, "text_rule", "low"
    if s in VALIDATED_DESCRIPTOR_DECISIONS:
        return VALIDATED_DESCRIPTOR_DECISIONS[s], "validated_descriptor_rule", "very_high"
    if contains_any(s, TARGET_TERMS):
        return "target", "text_rule", "high"
    if contains_any(s, CAVITY_TERMS):
        return "cavity_or_bed", "text_rule", "high"
    if contains_any(s, STRICT_STRUCTURE_TERMS):
        return "other_structure", "text_rule", "high"
    if contains_any(s, STRICT_LESION_TERMS):
        return "lesion", "text_rule", "high"
    if re.match(r"^\d+[_\-]", s):
        return "lesion", "text_rule", "high"
    if contains_any(s, LESION_LOCATION_TERMS) or contains_any(s, LESION_MODIFIER_TERMS):
        return "lesion", "text_rule", "medium"
    return None, "text_rule", "low"


def morphology_rule_classification(s: str, f: Dict[str, Any]) -> Tuple[Optional[str], str, str]:
    voxel_count  = f.get("mask_voxel_count", 0)  or 0
    volume_cc    = f.get("mask_volume_cc",   0.0) or 0.0
    fill_ratio   = f.get("mask_fill_ratio_in_bbox", 0.0) or 0.0
    n_components = f.get("n_components", None)
    n_slices     = f.get("n_slices_involved", 0) or 0

    if voxel_count == 0:
        return "other_structure", "morphology_rule", "medium"
    if volume_cc > 3.0 and n_slices > 40:
        return "other_structure", "morphology_rule", "high"
    if n_slices > 40 and volume_cc > 1.0 and fill_ratio < 0.4:
        return "other_structure", "morphology_rule", "high"
    if voxel_count <= VERY_SMALL_VOXELS:
        return "lesion", "morphology_rule", "medium"
    if voxel_count <= SMALL_VOXELS and fill_ratio < 0.9:
        return "lesion", "morphology_rule", "medium"
    if n_components is not None and n_components > 1 and voxel_count < 1000:
        return "lesion", "morphology_rule", "low"
    if volume_cc <= 1.5 and n_slices <= 30 and 0.15 <= fill_ratio <= 0.8:
        if contains_any(s, LESION_LOCATION_TERMS) or contains_any(s, LESION_MODIFIER_TERMS):
            return "lesion", "morphology_rule", "medium"
    return None, "morphology_rule", "low"


def final_category_decision(mask_desc: str, mask_features: Dict[str, Any]) -> Tuple[str, str, str]:
    s = normalize_text(mask_desc)
    cat, src, conf = text_rule_classification(s)
    if cat is not None:
        return cat, src, conf
    if mask_features.get("mask_read_ok") is not True:
        return "manual_review_required", "mask_unavailable_for_morphology", "low"
    cat, src, conf = morphology_rule_classification(s, mask_features)
    if cat is not None:
        return cat, src, conf
    return "manual_review_required", "fallback", "low"


# =========================================================
# 7) MAIN
# =========================================================

def main() -> None:
    metadata_dir    = PROJECT_ROOT / "metadata"
    processed_dir   = PROJECT_ROOT / "data" / "processed"
    logs_dir        = PROJECT_ROOT / "logs"
    checkpoints_dir = PROJECT_ROOT / "checkpoints"

    for p in [metadata_dir, processed_dir, logs_dir, checkpoints_dir]:
        p.mkdir(parents=True, exist_ok=True)

    state_path = checkpoints_dir / "pipeline_state.json"
    log_file   = logs_dir / f"{STEP_NAME}.log"
    logger     = setup_logger(log_file)
    state      = StateManager(state_path)

    logger.info("=" * 80)
    logger.info("Starting step 03 segmentation harmonization")

    if not state.is_step_done("step_02_data_audit"):
        raise RuntimeError("step_02 must be completed before step_03.")
    if not INPUT_CANONICAL.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_CANONICAL}")

    df = pd.read_csv(INPUT_CANONICAL)
    logger.info("Loaded canonical dataset: %s", df.shape)

    for col in ["ER", "PR", "HER2"]:
        if col in df.columns:
            df[f"{col}_bin"] = df[col].apply(normalize_label_pm)

    best_desc = df.apply(
        lambda row: choose_best_segmentation_text(
            row.get("Segmentation_Name"), row.get("matched_mask_file"), row.get("mask_path"),
        ), axis=1, result_type="expand",
    )
    best_desc.columns = ["mask_descriptor_raw", "mask_descriptor_source"]
    df = pd.concat([df, best_desc], axis=1)

    df["mask_descriptor_canonical"] = df["mask_descriptor_raw"].apply(canonicalize_mask_core)
    if "Segmentation_Name" in df.columns:
        df["segmentation_name_clean"] = df["Segmentation_Name"].apply(normalize_text)
    else:
        df["segmentation_name_clean"] = ""

    init_results = df["mask_descriptor_canonical"].apply(text_rule_classification)
    df["initial_mask_category"]        = init_results.apply(lambda x: x[0] if x[0] is not None else "unclear")
    df["initial_resolution_source"]    = init_results.apply(lambda x: x[1])
    df["initial_resolution_confidence"] = init_results.apply(lambda x: x[2])

    # FIX I-14: Check NIfTI library availability before attempting mask extraction
    _nifti_available = False
    if FORCE_MASK_FEATURE_EXTRACTION:
        try:
            import nibabel as _nib
            _nifti_available = True
        except ImportError:
            try:
                import SimpleITK as _sitk
                _nifti_available = True
            except ImportError:
                logger.warning(
                    "FORCE_MASK_FEATURE_EXTRACTION=True but neither nibabel nor SimpleITK "
                    "is installed. Mask morphology features will be skipped. "
                    "Install with: pip install nibabel --break-system-packages"
                )
        if not _nifti_available:
            logger.warning("Setting FORCE_MASK_FEATURE_EXTRACTION to False for this run.")
    # Local override: never mutate the module-level FORCE_MASK_FEATURE_EXTRACTION global.
    _effective_force_mask = FORCE_MASK_FEATURE_EXTRACTION and _nifti_available

    valid_mask_rows    = int(df["mask_abs_path"].dropna().astype(str).str.strip().replace("", np.nan).dropna().shape[0]) if "mask_abs_path" in df.columns else 0
    unique_mask_paths  = int(df["mask_abs_path"].dropna().astype(str).str.strip().replace("", np.nan).dropna().nunique()) if "mask_abs_path" in df.columns else 0
    logger.info("Rows with non-empty mask paths: %d | Unique: %d", valid_mask_rows, unique_mask_paths)

    feature_records       = []
    final_categories      = []
    final_sources         = []
    final_confidences     = []
    final_review_flags    = []
    final_used_morphology = []

    mask_feature_cache: Dict[str, Dict[str, Any]] = {}
    cache_hits = cache_misses = early_text_resolutions = actual_mask_reads = 0
    rows_total = int(df.shape[0])

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        desc      = row.get("mask_descriptor_canonical", "") or ""
        desc_norm = normalize_text(desc)
        mask_path = row.get("mask_abs_path") if "mask_abs_path" in df.columns else None
        need_mask = pd.notna(mask_path) and str(mask_path).strip() != ""

        if (i % PROGRESS_LOG_EVERY == 0) or (i == rows_total):
            logger.info("Progress: %d/%d | cache_hits=%d | early_text=%d | mask_reads=%d",
                        i, rows_total, cache_hits, early_text_resolutions, actual_mask_reads)

        mask_features = build_empty_mask_feature_record()
        used_morphology = False

        def _read_and_cache(mpath_str: str) -> None:
            nonlocal actual_mask_reads, cache_hits, cache_misses
            if ENABLE_MASK_FEATURE_CACHE and mpath_str in mask_feature_cache:
                mask_features.update(mask_feature_cache[mpath_str])
                cache_hits += 1
            else:
                actual_mask_reads += 1
                try:
                    mask_arr, spacing = load_mask_array(mpath_str)
                    computed = compute_mask_features(mask_arr, spacing)
                    mask_features.update(computed)
                    mask_features["mask_read_ok"] = True
                except Exception as e:
                    mask_features["mask_read_ok"]    = False
                    mask_features["mask_read_error"] = str(e)
                if ENABLE_MASK_FEATURE_CACHE:
                    mask_feature_cache[mpath_str] = dict(mask_features)
                    cache_misses += 1

        if _effective_force_mask and need_mask:
            _read_and_cache(str(mask_path).strip())

        if ENABLE_EARLY_TEXT_RESOLUTION:
            text_cat, text_src, text_conf = text_rule_classification(desc_norm)
            if text_cat is not None:
                final_cat, final_src, final_conf = text_cat, text_src, text_conf
                early_text_resolutions += 1
            else:
                if need_mask and not _effective_force_mask:
                    _read_and_cache(str(mask_path).strip())
                final_cat, final_src, final_conf = final_category_decision(desc, mask_features)
                used_morphology = (final_src == "morphology_rule")
        else:
            if need_mask and not _effective_force_mask:
                _read_and_cache(str(mask_path).strip())
            final_cat, final_src, final_conf = final_category_decision(desc, mask_features)
            used_morphology = (final_src == "morphology_rule")

        feature_records.append(dict(mask_features))
        final_categories.append(final_cat)
        final_sources.append(final_src)
        final_confidences.append(final_conf)
        final_review_flags.append(final_cat == "manual_review_required")
        final_used_morphology.append(int(used_morphology))

    feature_df = pd.DataFrame(feature_records)
    df = pd.concat([df, feature_df], axis=1)

    df["final_mask_category"]         = final_categories
    df["final_resolution_source"]     = final_sources
    df["final_resolution_confidence"] = final_confidences
    df["needs_manual_review_final"]   = final_review_flags
    df["used_morphology_resolution"]  = final_used_morphology

    df["is_true_lesion_candidate_final"]    = (df["final_mask_category"] == "lesion").astype(int)
    df["is_target_candidate_final"]         = (df["final_mask_category"] == "target").astype(int)
    df["is_cavity_or_bed_candidate_final"]  = (df["final_mask_category"] == "cavity_or_bed").astype(int)
    df["is_other_structure_candidate_final"]= (df["final_mask_category"] == "other_structure").astype(int)
    df["is_unresolved_candidate_final"]     = (df["final_mask_category"] == "manual_review_required").astype(int)

    label_bin_cols = [c for c in ["ER_bin", "PR_bin", "HER2_bin"] if c in df.columns]
    df["has_any_valid_label_final"] = df[label_bin_cols].notna().any(axis=1).astype(int) if label_bin_cols else 0

    df["passes_mask_resolution_guard"] = np.where(
        df["used_morphology_resolution"] == 1,
        (df["mask_read_ok"] == True).astype(int),
        1,
    )
    df["is_analysis_candidate_final"] = (
        (df["is_true_lesion_candidate_final"]    == 1)
        & (df["has_any_valid_label_final"]        == 1)
        & (df["needs_manual_review_final"]        == False)
        & (df["passes_mask_resolution_guard"]     == 1)
    ).astype(int)

    radiomics_cols = [c for c in df.columns if is_radiomics_column(c)]

    summary = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "n_unique_patients": int(df["patient_base"].nunique(dropna=True)) if "patient_base" in df.columns else None,
        "n_unique_cases": int(df["case_id"].nunique(dropna=True)) if "case_id" in df.columns else None,
        "n_radiomics_columns": len(radiomics_cols),
        "initial_mask_category_counts": df["initial_mask_category"].value_counts(dropna=False).to_dict(),
        "final_mask_category_counts":   df["final_mask_category"].value_counts(dropna=False).to_dict(),
        "needs_manual_review_final_count": int(df["needs_manual_review_final"].sum()),
        "analysis_candidate_final_count":  int(df["is_analysis_candidate_final"].sum()),
        "label_non_null_counts": {
            "ER_bin":   int(df["ER_bin"].notna().sum())   if "ER_bin"   in df.columns else 0,
            "PR_bin":   int(df["PR_bin"].notna().sum())   if "PR_bin"   in df.columns else 0,
            "HER2_bin": int(df["HER2_bin"].notna().sum()) if "HER2_bin" in df.columns else 0,
        },
        "mask_feature_coverage": {
            "mask_read_ok_count":    int((df["mask_read_ok"] == True).sum())  if "mask_read_ok" in df.columns else 0,
            "mask_read_failed_count":int((df["mask_read_ok"] == False).sum()) if "mask_read_ok" in df.columns else 0,
        },
        "performance_summary": {
            "early_text_resolutions":    early_text_resolutions,
            "actual_mask_reads":         actual_mask_reads,
            "cache_hits":                cache_hits,
            "cache_misses":              cache_misses,
            "force_mask_extraction":     FORCE_MASK_FEATURE_EXTRACTION,
        },
        "validated_descriptor_decisions_used": VALIDATED_DESCRIPTOR_DECISIONS,
        "fixes_applied": {
            "tum_tumo_added_to_strict_lesion_terms": True,
            "op_to_optic_replacement_removed":       True,
            "radiomics_prefix_anchoring_consistent": True,
        },
    }

    audit_columns = [c for c in [
        "FilenamePrefix", "ID", "case_id", "patient_base", "visit_index",
        "Segmentation_Name", "mask_path", "mask_abs_path", "matched_mask_file",
        "mask_descriptor_source", "mask_descriptor_raw", "mask_descriptor_canonical",
        "initial_mask_category", "final_mask_category", "final_resolution_source",
        "final_resolution_confidence", "used_morphology_resolution",
        "needs_manual_review_final", "mask_read_ok", "mask_read_error",
        "mask_voxel_count", "mask_volume_cc", "n_slices_involved",
        "mask_fill_ratio_in_bbox", "passes_mask_resolution_guard",
        "is_analysis_candidate_final",
        "ER", "PR", "HER2", "ER_bin", "PR_bin", "HER2_bin",
    ] if c in df.columns]

    audit_df = df[audit_columns].copy()
    audit_path = metadata_dir / "step03_resolution_audit.csv"
    audit_df.to_csv(audit_path, index=False, encoding="utf-8-sig")

    unresolved_path = metadata_dir / "step03_unresolved_only.csv"
    audit_df[audit_df["needs_manual_review_final"] == True].to_csv(unresolved_path, index=False, encoding="utf-8-sig")

    descriptor_summary = (
        df.groupby(["mask_descriptor_canonical", "final_mask_category"], dropna=False)
        .size().reset_index(name="n_rows")
        .sort_values(["final_mask_category", "n_rows", "mask_descriptor_canonical"], ascending=[True, False, True])
    )
    descriptor_summary.to_csv(metadata_dir / "step03_descriptor_summary.csv", index=False, encoding="utf-8-sig")

    full_csv    = processed_dir / "analysis_ready_step03_full.csv"
    full_parquet = processed_dir / "analysis_ready_step03_full.parquet"
    df.to_csv(full_csv, index=False, encoding="utf-8-sig")
    if SAVE_PARQUET_IF_AVAILABLE:
        try:
            df.to_parquet(full_parquet, index=False)
        except Exception as e:
            logger.warning("Skipping parquet: %s", e)

    lesion_only_csv = lesion_only_parquet = None
    if BUILD_LESION_ONLY_SUBSET:
        lesion_df        = df[df["is_analysis_candidate_final"] == 1].copy()
        lesion_only_csv  = processed_dir / "analysis_ready_step03_lesion_only.csv"
        lesion_only_parquet = processed_dir / "analysis_ready_step03_lesion_only.parquet"
        lesion_df.to_csv(lesion_only_csv, index=False, encoding="utf-8-sig")
        if SAVE_PARQUET_IF_AVAILABLE:
            try:
                lesion_df.to_parquet(lesion_only_parquet, index=False)
            except Exception as e:
                logger.warning("Skipping lesion parquet: %s", e)
        summary["lesion_only_rows"]            = int(lesion_df.shape[0])
        summary["lesion_only_unique_patients"] = int(lesion_df["patient_base"].nunique(dropna=True)) if "patient_base" in lesion_df.columns else None
        summary["lesion_only_unique_cases"]    = int(lesion_df["case_id"].nunique(dropna=True)) if "case_id" in lesion_df.columns else None

    summary_path = metadata_dir / "step03_summary.json"
    save_json(summary, summary_path)

    artifacts = {
        "step03_summary_json":              str(summary_path),
        "step03_resolution_audit_csv":      str(audit_path),
        "step03_unresolved_only_csv":       str(unresolved_path),
        "step03_descriptor_summary_csv":    str(metadata_dir / "step03_descriptor_summary.csv"),
        "analysis_ready_step03_full_csv":   str(full_csv),
        "step_03_log_file":                 str(log_file),
    }
    if lesion_only_csv:
        artifacts["analysis_ready_step03_lesion_only_csv"] = str(lesion_only_csv)

    state.mark_step_done(STEP_NAME, artifacts=artifacts)
    state.add_note("Step 03 segmentation harmonization completed successfully.")
    logger.info("Step 03 completed. Analysis candidates: %d", summary.get("analysis_candidate_final_count", 0))
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
