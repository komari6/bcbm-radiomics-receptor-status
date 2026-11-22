#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 13b — Methods pipeline figure (manuscript Figure 1)
=========================================================

Purpose
-------
Draws the leakage-controlled workflow diagram. Previously this figure was a hand-drawn
asset with no source, which (a) blocked the paper's reproducibility claim and (b) let it
drift out of agreement with the code: the retired asset showed ComBat *inside* the
sklearn pipeline between the correlation filter and stability selection, and omitted the
missingness filter entirely.

The order drawn here is the order the code actually executes:
  ComBat is applied first, OUTSIDE the sklearn pipeline (step 05,
  `harmonize_pair_if_needed`, fit on the training fold and applied to held-out data),
  then the pipeline runs (step 05, `build_preprocessing_steps`):
      missingness -> imputer -> variance -> correlation -> stability -> scaler -> model

Outputs
-------
  reports/figures/step13_publication_package/fig_methods_pipeline.{png,pdf}

Usage
-----
  python src/reporting/step_13b_methods_figure.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

_PROJECT = Path(__file__).resolve().parents[2] / "bcbm_project"
OUTDIR = _PROJECT / "reports" / "figures" / "step13_publication_package"
FEATURE_DICT = _PROJECT / "metadata" / "step04_feature_dictionary.json"


def block_sizes():
    """Feature-block sizes, read from the dictionary rather than typed into the panel.

    These were hard-coded, and stayed at 75/25 after four mask-derived columns moved from the
    acquisition block to lesion burden -- so the first figure of the paper disagreed with
    Table 1, Table 2 and Table S9.
    """
    import json
    with open(FEATURE_DICT, encoding="utf-8") as fh:
        d = json.load(fh)
    return (len(d["radiomic_block_columns"]), len(d["burden_study_block_columns"]),
            len(d["acquisition_block_columns"]))

# Palette shared with the rest of the publication package.
C_DATA = "#4C9BD1"   # source data
C_HARM = "#1B9E77"   # harmonization
C_SPLIT = "#E0B84C"  # partitioning / feature selection
C_PREP = "#E8705F"   # in-fold preprocessing
C_MODEL = "#2B3A4A"  # model / output
C_OK = "#4CAF50"     # threshold step
C_GUARD = "#FDECEA"  # leakage guard
C_PANEL = "#EDF1F5"  # nested-CV panel

FONT = "DejaVu Sans"


def _contrast_text(facecolor) -> str:
    """White or near-black, whichever has the higher WCAG contrast ratio on `facecolor`."""
    r, g, b = mcolors.to_rgb(facecolor)
    lin = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    lum = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
    return "#FFFFFF" if (1.05 / (lum + 0.05)) >= ((lum + 0.05) / 0.05) else "#111111"


def box(ax, x, y, w, h, lines, fc, *, ec=None, lw=1.4, ls="-", fs=10.5, bold_first=True):
    """One rounded box. `lines` is a list of (text, is_bold) or plain strings."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004", facecolor=fc,
        edgecolor=ec or C_MODEL, linewidth=lw, linestyle=ls, zorder=2))
    if isinstance(lines, str):
        lines = [lines]
    norm = [(t, (i == 0 and bold_first)) if isinstance(t, str) else t for i, t in enumerate(lines)]
    n = len(norm)
    for i, (txt, bold) in enumerate(norm):
        yy = y + h - h * (i + 0.5) / n
        ax.text(x + w / 2, yy, txt, ha="center", va="center", zorder=3,
                fontsize=fs if bold else fs - 1.3, fontweight="bold" if bold else "normal",
                color=_contrast_text(fc), family=FONT)


def arrow(ax, x, y0, y1, *, color=C_MODEL, ls="-", lw=1.4):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls,
                                shrinkA=0, shrinkB=0), zorder=1)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.9, 12.1))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.06); ax.axis("off")   # headroom above 1.0 for the title block

    ax.text(0.5, 1.035, "Leakage-Controlled Radiomics Pipeline", ha="center", va="center",
            fontsize=17, fontweight="bold", color="#111111", family=FONT)
    ax.text(0.5, 0.996, "Every label-dependent step is fit inside the training fold only",
            ha="center", va="center", fontsize=10.5, style="italic", color="#333333", family=FONT)

    box(ax, 0.16, 0.916, 0.68, 0.038,
        ["Public BCBM dataset — 165 patients",
         "T1 post-contrast MRI  ·  expert-reviewed masks  ·  107 PyRadiomics features"], C_DATA)
    arrow(ax, 0.5, 0.916, 0.899)

    # This box is PRE-SPLIT, so it must not carry the harmonization colour: the only
    # harmonization in this pipeline is the fold-internal ComBat further down. What happens
    # here is mask/label canonicalization (step 03) and patient-level aggregation (step 04),
    # neither of which looks at labels or at held-out data.
    box(ax, 0.13, 0.860, 0.74, 0.038,
        ["Mask/label canonicalization → lesion-only table (1,841 lesions)",
         "Patient-level aggregation → %d radiomic  ·  %d lesion-burden/study  ·  %d acquisition"
         % block_sizes()], C_DATA)
    arrow(ax, 0.5, 0.860, 0.843)

    box(ax, 0.20, 0.804, 0.60, 0.038,
        ["Stratified PATIENT-LEVEL split", "Train 102  ·  Tuning 20  ·  Test 17"], C_SPLIT)
    box(ax, 0.825, 0.800, 0.165, 0.046,
        [("Leakage guard:", False), ("no patient's lesions", False), ("span two splits", False)],
        C_GUARD, ec="#C0392B", ls="--", lw=1.3, fs=8.4, bold_first=False)
    ax.annotate("", xy=(0.822, 0.823), xytext=(0.80, 0.823),
                arrowprops=dict(arrowstyle="-|>", color="#C0392B", lw=1.3))

    # locked-test callout
    box(ax, 0.012, 0.700, 0.165, 0.052,
        [("LOCKED TEST", True), ("(17 patients)", False), ("untouched until", False),
         ("final evaluation", False)], "#FFFFFF", ec="#C0392B", lw=1.6, fs=9)
    ax.annotate("", xy=(0.175, 0.786), xytext=(0.20, 0.804),
                arrowprops=dict(arrowstyle="-|>", color="#C0392B", lw=1.3))
    ax.annotate("", xy=(0.28, 0.246), xytext=(0.095, 0.700),
                arrowprops=dict(arrowstyle="-|>", color="#C0392B", lw=1.2, linestyle="--"))

    arrow(ax, 0.5, 0.804, 0.789)

    # ---- nested CV panel ----
    ax.add_patch(FancyBboxPatch((0.20, 0.286), 0.66, 0.500, boxstyle="round,pad=0.004",
                                facecolor=C_PANEL, edgecolor=C_MODEL, linewidth=1.6, zorder=0))
    ax.text(0.53, 0.768, "Repeated nested cross-validation", ha="center",
            fontsize=13, fontweight="bold", color="#111111", family=FONT)
    ax.text(0.53, 0.752, "5 outer × 3 inner  ·  10 repetitions", ha="center",
            fontsize=9.6, fontweight="bold", color="#111111", family=FONT)
    ax.text(0.53, 0.737, "The outer partition is redrawn at every repetition",
            ha="center", fontsize=8.8, style="italic", color="#333333", family=FONT)
    ax.text(0.222, 0.520, "outer ×5\n(estimate AUROC)", rotation=90, ha="center", va="center",
            fontsize=8.6, color="#333333", family=FONT)
    ax.text(0.845, 0.520, "inner ×3\n(select model)", rotation=270, ha="center", va="center",
            fontsize=8.6, color="#333333", family=FONT)

    box(ax, 0.255, 0.700, 0.555, 0.028, ["Within EACH training fold only:"], "#FFFFFF", fs=10.5)

    # The executed order. ComBat is first and sits OUTSIDE the sklearn pipeline.
    steps = [
        ("Location-scale ComBat-style harmonization  (no empirical Bayes;\nfit on train → apply to held-out)", C_HARM),
        ("Missingness filter  (drop features missing in > 40% of patients)", C_PREP),
        ("Median imputation  (SimpleImputer)", C_PREP),
        ("Variance filter", C_PREP),
        ("Correlation filter  (|r| > 0.88)", C_PREP),
        # "bootstraps" was wrong: the implementation draws 50 subsamples of 80% WITHOUT
        # replacement (rng.choice(..., replace=False)), which is what the Methods describe.
        ("Stability selection  (50 subsamples of 80%, ≤ 20 features)", C_SPLIT),
        ("Feature scaling  (RobustScaler; StandardScaler for the calibrated SVM)", C_PREP),
        ("Classifier  (elastic-net / GBT / extra-trees / SVM / PLS)", C_MODEL),
    ]
    y, h, gap = 0.663, 0.030, 0.0075
    for i, (txt, col) in enumerate(steps):
        nl = txt.count("\n")
        hh = h + (0.012 if nl else 0)
        box(ax, 0.255, y - hh, 0.555, hh, [(txt, False)], col, fs=9.4, bold_first=False)
        if i < len(steps) - 1:
            arrow(ax, 0.5325, y - hh, y - hh - gap, lw=1.1)
        y = y - hh - gap

    arrow(ax, 0.5325, y, y - 0.020)
    box(ax, 0.255, y - 0.050, 0.555, 0.030,
        ["Youden threshold on OOF predictions  (never on test)"], C_OK, fs=10)

    arrow(ax, 0.5325, 0.286, 0.262)
    box(ax, 0.16, 0.224, 0.68, 0.036,
        ["Refit on Train + Tuning → evaluate LOCKED TEST once"], C_MODEL, fs=11.5)

    # branches
    ax.annotate("", xy=(0.27, 0.176), xytext=(0.40, 0.224),
                arrowprops=dict(arrowstyle="-|>", color=C_MODEL, lw=1.3))
    ax.annotate("", xy=(0.79, 0.176), xytext=(0.66, 0.224),
                arrowprops=dict(arrowstyle="-|>", color=C_MODEL, lw=1.3))
    arrow(ax, 0.5, 0.224, 0.132)

    box(ax, 0.025, 0.128, 0.42, 0.048,
        [("Deep models (parallel benchmark)", False),
         ("Attention-MIL · 2.5-D CNN · Hybrid fusion", False),
         ("(checkpoint chosen on the tuning set, not test)", False)],
        "#FFFFFF", fs=8.8, bold_first=False)
    box(ax, 0.555, 0.128, 0.42, 0.048,
        [("External PROXY cohort (OpenBTAI)", False),
         ("applied ONCE; no transform refit on it", False),
         ("(subtype-derived surrogate labels)", False)],
        "#FFFFFF", ec="#888888", ls="--", fs=8.8, bold_first=False)

    box(ax, 0.075, 0.040, 0.85, 0.060,
        [("Statistical evaluation", True),
         ("Primary: mean AUROC ± SD over 10 partitions  ·  cross-fitted AUROC, patient bootstrap CI (2000×)", False),
         ("Secondary: locked test (n = 16–17)  ·  label permutation  ·  Benjamini–Hochberg FDR per target", False)],
        "#EAF2FA", fs=9.6)

    # Legend. x positions are set per entry rather than on a fixed pitch: "Preprocessing
    # (in-fold)" is far wider than the rest and ran into the next swatch on an even pitch.
    for x, lab, col in [(0.012, "Preprocessing (in-fold)", C_PREP),
                        (0.250, "Harmonization (in-fold)", C_HARM),
                        (0.487, "Feature selection", C_SPLIT),
                        (0.665, "Model / output", C_MODEL),
                        (0.838, "Leakage guard", C_GUARD)]:
        ax.add_patch(FancyBboxPatch((x, 0.006), 0.024, 0.015, boxstyle="round,pad=0.002",
                                    facecolor=col, edgecolor="#555555", linewidth=0.8))
        ax.text(x + 0.030, 0.0135, lab, va="center", fontsize=8.2, color="#111111", family=FONT)

    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"fig_methods_pipeline.{ext}", dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print("wrote", OUTDIR / "fig_methods_pipeline.png")


if __name__ == "__main__":
    main()
