# -*- coding: utf-8 -*-
"""Tile the supplementary figures into multi-panel composites.

A supplement held to a 12-page limit cannot carry twenty one-per-page figures. (Where no such
limit applies, tools/render_print_supplement.py prints each figure on its own page instead.) Nothing is dropped: every panel of every
original figure appears here at full extent, tiled into six composites, and the original files stay
in the step-13 package. Panels are never relettered -- several of the sources already carry their
own (A)/(B) labels -- so the merged legends identify parts by position instead.

A row's panels are scaled to a common height and laid out so the row fills the text width; rows are
stacked. Only downscaling ever happens (the sources are ~500-600 dpi for their printed width), and
`width` on a single-panel row holds a tall panel back from filling the column.

    python tools/build_supp_composites.py
"""
import io
import os
import sys

from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "bcbm_project", "reports", "figures", "step13_publication_package")
OUT = SRC                                  # composites live beside their sources
TEXT_IN = 6.5                              # printed width: 8.5 in page less 1 in margins
DPI = 600
GAP_IN = 0.06

# tag -> rows of panels; a row is a list of source files, or (file, width_fraction) alone on its row
COMPOSITES = [
    ("fig_SC1_cohort_labels_burden.png", [
        ["fig_01_cohort_consort.png", "fig_S1_lesion_category_breakdown.png"],
        ["fig_02_label_distribution.png", "fig_S2_patient_lesion_burden.png"],
    ]),
    ("fig_SC2_acquisition_quality.png", [
        ["fig_03_scanner_audit.png"],
        ["fig_S3_field_strength_split_balance.png", "fig_S4_feature_missingness.png"],
    ]),
    ("fig_SC3_discrimination_operating_point.png", [
        ["fig_06_roc_curves.png"],
        ["fig_09_confusion_matrices.png"],
    ]),
    ("fig_SC4_feature_importance.png", [
        [("fig_04_feature_importance_heatmap.png", 0.92)],
        ["fig_05_radiomics_violin.png"],
    ]),
    ("fig_SC5_model_families.png", [
        ["fig_10_model_comparison_heatmap.png"],
        [("fig_11_mil_attention_schematic.png", 0.72)],
    ]),
    ("fig_SC6_stability_null.png", [
        [("fig_14_feature_stability.png", 0.60)],
        ["fig_permutation_null.png"],
    ]),
    ("fig_SC7_optimism_transfer.png", [
        ["fig_optimism_generalization.png"],
        ["fig_external_feature_transfer.png"],
    ]),
]


def on_white(im):
    """Flatten to RGB on white so transparent margins do not print grey."""
    im = im.convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, im).convert("RGB")


def lay_out(rows, width_px, gap_px):
    """[(image, x, y)], total height -- rows scaled to a common height, filling the width."""
    placed, y = [], 0
    for row in rows:
        panels = [(p, 1.0) if isinstance(p, str) else p for p in row]
        ims = [on_white(Image.open(os.path.join(SRC, f))) for f, _ in panels]
        if len(panels) == 1:
            w = int(round(width_px * panels[0][1]))
            im = ims[0]
            h = int(round(w * im.height / im.width))
            placed.append((im.resize((w, h), Image.LANCZOS), (width_px - w) // 2, y))
            y += h + gap_px
            continue
        # common height H such that the scaled widths plus the gaps fill the row
        avail = width_px - gap_px * (len(ims) - 1)
        h = int(round(avail / sum(im.width / im.height for im in ims)))
        x = 0
        for im in ims:
            w = int(round(h * im.width / im.height))
            placed.append((im.resize((w, h), Image.LANCZOS), x, y))
            x += w + gap_px
        y += h + gap_px
    return placed, max(0, y - gap_px)


def main():
    width_px = int(round(TEXT_IN * DPI))
    gap_px = int(round(GAP_IN * DPI))
    for name, rows in COMPOSITES:
        placed, height = lay_out(rows, width_px, gap_px)
        sheet = Image.new("RGB", (width_px, height), (255, 255, 255))
        for im, x, y in placed:
            sheet.paste(im, (x, y))
        path = os.path.join(OUT, name)
        sheet.save(path, "PNG", optimize=True)
        panels = sum(len(r) for r in rows)
        print("%-46s %5dx%-5d  %d panels, %.2f in tall at %.1f in wide"
              % (name, width_px, height, panels, height / DPI, TEXT_IN))
    return 0


if __name__ == "__main__":
    sys.exit(main())
