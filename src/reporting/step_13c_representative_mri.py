"""
step_13c_representative_mri.py — main Figure 2: representative lesion montage.

WHY THIS FILE EXISTS
--------------------
Figure 2 was originally produced on 2026-06-19 by a one-off script that was never committed
(the PNG's metadata shows Matplotlib 3.11.0, and the project's development log records it as
not part of run_pipeline.py). The manuscript's
Data/Code availability statement therefore carried an exception for Figure 2, and the selection
procedure could not be documented. This script reinstates the procedure as
documented, deterministic code so the figure is reproducible like every other one.

SELECTION PROCEDURE (the answer to "how were these lesions chosen?")
--------------------------------------------------------------------
1. Start from the analysis-eligible lesion table (1,841 lesions, 139 patients).
2. Keep only DISPLAY-ELIGIBLE lesions: mask volume >= MIN_DISPLAY_CC. Sub-half-cc lesions are a
   few voxels across and are unreadable at print scale; including them would make "the median
   lesion" a black square. This filter defines the candidate pool (~337 lesions) that the
   original figure's caption referred to.
3. Form six groups: ER-negative, ER-positive, HER2-negative, HER2-positive, 1.5 T, 3.0 T.
4. Within each group, compute the MEDIAN candidate volume and rank lesions by absolute distance
   from it. Walk that ranking and take the first two lesions whose patient has not already been
   used anywhere in the figure, so all twelve panels come from twelve different patients. This
   constraint displaces three panels away from the strict median-nearest lesion: HER2+ (the two
   nearest belong to patient 107, already shown as ER-negative) and the second 3.0-T panel (the
   four nearest belong to patients 96, 73 and 9). The legend states the constraint for this reason.
5. Render each chosen lesion on the axial slice where its mask has the largest cross-sectional
   area, cropped to the lesion's bounding box plus a fixed margin, with the expert mask outline
   overlaid.

The rule is median-within-the-candidate-pool, NOT median over all 1,841 lesions: with the full
set the median lesion is ~0.05-0.27 cc and cannot be displayed. Ties and ordering are resolved
deterministically (stable sort, fixed group order), so the figure is byte-reproducible.

Usage:  python src/reporting/step_13c_representative_mri.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[2] / "bcbm_project"
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = Path("./bcbm_project").resolve()  # legacy cwd-relative fallback, as in step_13
LESION_CSV = PROJECT_ROOT / "data" / "processed" / "analysis_ready_step03_lesion_only.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "step13_publication_package"
OUT_STEM = "fig_representative_mri"

MIN_DISPLAY_CC = 0.5      # display-eligibility threshold; defines the candidate pool
CROP_MARGIN_VOX = 22      # voxels of context around the lesion bounding box
NEG_COLOR = "#B3261E"
POS_COLOR = "#2F3E55"
CONTOUR_COLOR = "#FFC300"

ROWS = [
    ("ER status", "ER_bin", ("ER−", "ER+")),
    ("HER2 status", "HER2_bin", ("HER2−", "HER2+")),
    ("Field strength", "_fs", ("1.5 T", "3.0 T")),
]


def setup_logger() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger("step_13c")


def load_candidates(logger: logging.Logger) -> pd.DataFrame:
    d = pd.read_csv(LESION_CSV, low_memory=False)
    d = d.copy()
    d["cc"] = pd.to_numeric(d["mask_volume_mm3"], errors="coerce") / 1000.0
    d["_fs"] = (d["Magnetic_Field_Strength_ID"].astype(str)
                .str.extract(r"([\d.]+)")[0].astype(float))
    for c in ("ER_bin", "HER2_bin"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    logger.info("analysis-eligible lesions: %d (%d patients)", len(d), d["patient_base"].nunique())
    cand = d[d["cc"] >= MIN_DISPLAY_CC].copy()
    cand = cand[cand["image_abs_path"].map(lambda p: Path(str(p)).exists())]
    cand = cand[cand["mask_abs_path"].map(lambda p: Path(str(p)).exists())]
    logger.info("display-eligible candidates (>= %.2f cc, files present): %d",
                MIN_DISPLAY_CC, len(cand))
    return cand


def pick_two(cand: pd.DataFrame, mask: pd.Series, used_patients: set, logger: logging.Logger):
    """Two lesions nearest the group's median volume, from patients not already used."""
    grp = cand[mask].dropna(subset=["cc"])
    if grp.empty:
        return []
    med = float(grp["cc"].median())
    ranked = grp.assign(_d=(grp["cc"] - med).abs()).sort_values(
        ["_d", "cc", "patient_base"], kind="stable")
    chosen = []
    for _, r in ranked.iterrows():
        if r["patient_base"] in used_patients:
            continue
        used_patients.add(r["patient_base"])
        chosen.append(r)
        if len(chosen) == 2:
            break
    logger.info("  group median %.2f cc | n=%d | chose %s cc from patients %s",
                med, len(grp), [round(float(c["cc"]), 1) for c in chosen],
                [int(c["patient_base"]) for c in chosen])
    return chosen


def best_axial_slice(mask_arr: np.ndarray) -> int:
    areas = mask_arr.reshape(-1, mask_arr.shape[-1]).sum(axis=0)
    return int(np.argmax(areas))


def draw_panel(ax, row, label, color, logger):
    img = nib.load(str(row["image_abs_path"])).get_fdata()
    msk = np.asarray(nib.load(str(row["mask_abs_path"])).get_fdata() > 0, dtype=np.uint8)
    if img.shape != msk.shape:
        logger.warning("  shape mismatch for %s; skipping panel", row["FilenamePrefix"])
        ax.axis("off")
        return
    k = best_axial_slice(msk)
    im2, mk2 = img[:, :, k], msk[:, :, k]
    ys, xs = np.where(mk2 > 0)
    if ys.size == 0:
        ax.axis("off")
        return
    y0 = max(0, ys.min() - CROP_MARGIN_VOX); y1 = min(im2.shape[0], ys.max() + CROP_MARGIN_VOX)
    x0 = max(0, xs.min() - CROP_MARGIN_VOX); x1 = min(im2.shape[1], xs.max() + CROP_MARGIN_VOX)
    crop_i, crop_m = im2[y0:y1, x0:x1], mk2[y0:y1, x0:x1]
    vmax = np.percentile(crop_i[crop_i > 0], 99.5) if (crop_i > 0).any() else 1.0
    ax.imshow(np.rot90(crop_i), cmap="gray", vmin=0, vmax=vmax, interpolation="bilinear")
    # contour without scikit-image: matplotlib's own contour at the 0.5 level
    ax.contour(np.rot90(crop_m), levels=[0.5], colors=[CONTOUR_COLOR], linewidths=1.6)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(0.035, 0.965, label, transform=ax.transAxes, va="top", ha="left", fontsize=10,
            color="white", bbox=dict(facecolor=color, edgecolor="none", pad=2.4))
    ax.text(0.945, 0.035, f"{float(row['cc']):.1f} cc", transform=ax.transAxes,
            va="bottom", ha="right", fontsize=9, color="white")


def main() -> int:
    logger = setup_logger()
    cand = load_candidates(logger)
    used: set = set()
    picks = []
    for row_label, col, (neg_lab, pos_lab) in ROWS:
        logger.info("row: %s", row_label)
        if col == "_fs":
            masks = [(cand["_fs"] == 1.5), (cand["_fs"] == 3.0)]
        else:
            masks = [(cand[col] == 0), (cand[col] == 1)]
        left = pick_two(cand, masks[0], used, logger)
        right = pick_two(cand, masks[1], used, logger)
        picks.append((row_label, (neg_lab, left), (pos_lab, right)))

    fig, axes = plt.subplots(3, 4, figsize=(13.0, 10.2))
    fig.suptitle("Representative brain-metastasis lesions "
                 "(T1 post-contrast; yellow = expert segmentation)",
                 fontweight="bold", fontsize=13.5, y=0.985)
    for r, (row_label, (neg_lab, left), (pos_lab, right)) in enumerate(picks):
        for c in range(4):
            ax = axes[r][c]
            lab, color = (neg_lab, NEG_COLOR) if c < 2 else (pos_lab, POS_COLOR)
            group = left if c < 2 else right
            idx = c if c < 2 else c - 2
            if idx < len(group):
                draw_panel(ax, group[idx], lab, color, logger)
            else:
                ax.axis("off")
        axes[r][0].set_ylabel(row_label, fontsize=12, fontweight="bold")
        axes[r][0].yaxis.set_visible(True); axes[r][0].set_yticks([])
    fig.text(0.30, 0.930, "Negative", ha="center", fontsize=12.5, fontweight="bold", color=NEG_COLOR)
    fig.text(0.72, 0.930, "Positive", ha="center", fontsize=12.5, fontweight="bold", color=POS_COLOR)
    fig.text(0.5, 0.012,
             "Column headings apply to rows 1–2; row 3 contrasts 1.5 T and 3.0 T acquisitions. "
             "Public, de-identified data.", ha="center", fontsize=9.5, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.918])
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{OUT_STEM}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{OUT_STEM}.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", FIGURES_DIR / f"{OUT_STEM}.png")
    logger.info("patients used (must all differ): %s", sorted(int(p) for p in used))
    return 0


if __name__ == "__main__":
    sys.exit(main())
