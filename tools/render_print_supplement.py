# -*- coding: utf-8 -*-
"""Render the supporting-information figures (S1 File), one figure per page.

The earlier supplement packed its figures into seven composites to fit a page limit; four source
figures per 6.2-inch-wide image put their text at 4-5 pt in print. With no page limit on the
supporting information, each source figure stands on its own page (landscape or portrait,
whichever prints it larger) and is re-drawn at that printed size in Arial with no text below 8 pt,
exactly as tools/render_print_figures.py does for the article's figures.

The panels already shown in the article are not repeated: the cohort flow (Fig 1), the
permutation null with seed stability (Fig 7) and the optimism across 18 datasets (Fig 8).

Every figure is drawn by the pipeline's own function in step_13, from the stored result files; the
function's own save is intercepted, so nothing in the pipeline's figure package is overwritten.
The one exception is the external feature-transfer figure (step 18): its PCA coordinates are not
stored, so it cannot be redrawn without re-running that analysis; the stored image is used with its
printed title cropped, and it prints at about 8.5 pt on a landscape page.

    python tools/render_print_supplement.py [--out DIR]
"""
import importlib.util
import io
import json
import logging
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("prf", os.path.join(ROOT, "tools", "render_print_figures.py"))
prf = importlib.util.module_from_spec(spec)
sys.modules["prf"] = prf
spec.loader.exec_module(prf)
plt, Text, matplotlib = prf.plt, prf.Text, prf.matplotlib

# output folder: --out DIR on the command line, results/figures/print_supplement by default
OUT = os.path.join(ROOT, "results", "figures", "print_supplement")
if "--out" in sys.argv:
    OUT = os.path.abspath(sys.argv[sys.argv.index("--out") + 1])
PKG = os.path.join(ROOT, "bcbm_project", "reports", "figures", "step13_publication_package")
# printable area of the S1 File pages (A4, 0.79-in margins), less room for the legend under the figure
BOX = {"landscape": (10.0, 5.3), "portrait": (6.6, 7.9)}
FIGS = [("A", "fig_S1_lesion_category_breakdown"), ("B", "fig_02_label_distribution"),
        ("C", "fig_S2_patient_lesion_burden"), ("D", "fig_03_scanner_audit"),
        ("E", "fig_S3_field_strength_split_balance"), ("F", "fig_S4_feature_missingness"),
        ("G", "fig_06_roc_curves"), ("H", "fig_09_confusion_matrices"),
        ("I", "fig_04_feature_importance_heatmap"), ("J", "fig_05_radiomics_violin"),
        ("K", "fig_10_model_comparison_heatmap"), ("L", "fig_11_mil_attention_schematic"),
        ("M", "fig_14_feature_stability"), ("N", "fig_15_cross_stage_comparison")]
REPORT = {}


def text_overlaps(fig):
    """Pairs of printed labels whose boxes overlap by more than a hairline."""
    r = fig.canvas.get_renderer()
    # tick labels outside the view limits exist as Text objects but are never drawn; counting
    # them produced false overlaps (Fig G's "-0.2" under its "0.0"), so only drawn ones count
    all_ticks, drawn = set(), set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                all_ticks.update({id(tick.label1), id(tick.label2)})
            for tick in axis._update_ticks():
                drawn.update({id(tick.label1), id(tick.label2)})
    ts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()
          and (id(t) not in all_ticks or id(t) in drawn)]
    bbs = [t.get_window_extent(r) for t in ts]
    hits = []
    for i in range(len(ts)):
        for j in range(i + 1, len(ts)):
            a, b = bbs[i], bbs[j]
            w = min(a.x1, b.x1) - max(a.x0, b.x0)
            h = min(a.y1, b.y1) - max(a.y0, b.y0)
            if w <= 2 or h <= 2:
                continue
            # the same label drawn twice in one place (twin axes share the x ticks): not a collision
            same = ts[i].get_text() == ts[j].get_text() and w * h > 0.9 * min(a.width * a.height, b.width * b.height)
            # a rotated label's axis-aligned box is much larger than its glyphs; those pairs are
            # judged by eye instead (Fig K)
            rotated = any(abs(t.get_rotation()) % 90 > 1 for t in (ts[i], ts[j]))
            if not same and not rotated:
                hits.append((ts[i].get_text()[:25], ts[j].get_text()[:25]))
    return hits


def fig_A_breakdown(data):
    """The segmentation-category ring, drawn so the excluded categories sit under the excluded arc.
    The pipeline's version spread the five excluded categories round the whole inner circle, so
    "target" (803 of the 984 excluded) ran underneath the green lesion arc and read as lesions."""
    s3 = json.load(open(os.path.join(ROOT, "bcbm_project", "metadata", "step03_summary.json"), encoding="utf-8"))
    cats = s3["final_mask_category_counts"]
    n_les = len(data["step03"])
    total = int(sum(cats.values()))
    lbl = {"target": "Target", "other_structure": "Other structure", "cavity_or_bed": "Cavity/bed",
           "manual_review_required": "Manual review"}
    excl = {lbl.get(k, k): int(v) for k, v in cats.items() if k != "lesion"}
    excl["Lesion, not eligible"] = int(cats["lesion"]) - n_les
    assert sum(excl.values()) == total - n_les == 984 and n_les == 1841
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.pie([n_les, total - n_les], radius=1.0, colors=["#4CAF50", "#BDBDBD"], startangle=90, counterclock=False,
           labels=["Analysis-eligible lesions\n%d (%.1f%%)" % (n_les, 100 * n_les / total),
                   "Excluded structures\n%d (%.1f%%)" % (total - n_les, 100 * (total - n_les) / total)],
           wedgeprops=dict(width=0.28, edgecolor="white"), labeldistance=1.12)
    cols = ["#616161", "#8A8A8A", "#A8A8A8", "#C4C4C4", "#DADADA"]
    w, _ = ax.pie([n_les] + list(excl.values()), radius=0.70, startangle=90, counterclock=False,
                  colors=["none"] + cols, wedgeprops=dict(width=0.26, edgecolor="white"))
    w[0].set_linewidth(0)
    ax.legend(w[1:], ["%s: %d" % kv for kv in excl.items()], title="Excluded structures",
              loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.set_aspect("equal")
    return fig


def adjust(letter, fig):
    """Figure-specific repairs for defects the overlap checks found at print size."""
    if letter in ("E", "F", "I", "K", "N"):
        # single-plot figures carry a descriptive title that the legend in S1 File already gives
        for ax in fig.axes:
            ax.set_title("")
    if letter == "E":            # the 1.5T/3T key sat on top of the training bar: give the bars headroom
        ax = fig.axes[0]
        top = max(p.get_height() for p in ax.patches if isinstance(p, matplotlib.patches.Rectangle))
        ax.set_ylim(0, top * 1.28)
    if letter == "K":            # twelve two-line column labels collided once raised to 8 pt
        ax = fig.axes[0]
        for lab in ax.get_xticklabels():
            lab.set_rotation(35); lab.set_ha("right"); lab.set_rotation_mode("anchor")
    if letter == "J":            # the p value sat on top of the tallest violin: move it above the data
        for ax in fig.axes:
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + 0.18 * (hi - lo))
            for t in ax.texts:
                if t.get_text().startswith("p="):
                    t.set_text("p = " + t.get_text()[2:])
                    t.set_transform(ax.transAxes); t.set_position((0.5, 0.95)); t.set_va("top"); t.set_ha("center")
    if letter == "N":            # "Random" under the 0.5 tick collided with the tick label and axis
        for ax in fig.axes:      # title; the dashed chance line is described in the legend instead
            for t in list(ax.texts):
                if t.get_text().strip() == "Random":
                    t.remove()


def normalise(fig):
    """Drop the figure-level title, fit the drawing to the better page orientation, and raise every
    label to at least 8 pt at the printed size. Returns the orientation used."""
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.remove()
        fig._suptitle = None
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    fits = {o: min(1.0, w / bb.width, h / bb.height) for o, (w, h) in BOX.items()}
    orient = max(fits, key=fits.get)
    scale = 0.97 * fits[orient]
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text().strip():
            # Arial has no star glyph; the importance plot marks features with one
            t.set_fontfamily("DejaVu Sans" if "★" in t.get_text() else "Arial")
            t.set_fontsize(max(t.get_fontsize(), (prf.MIN_PT + 0.2) / scale))
    fig.set_size_inches(fig.get_size_inches() * scale)
    for t in fig.findobj(Text):
        t.set_fontsize(t.get_fontsize() * scale)
    for ln in fig.findobj(matplotlib.lines.Line2D):
        ln.set_linewidth(max(0.5, ln.get_linewidth() * scale))
        ln.set_markersize(max(2.0, ln.get_markersize() * scale))
    for p in fig.findobj(matplotlib.patches.Patch):
        p.set_linewidth(max(0.4, p.get_linewidth() * scale))
    for c in fig.findobj(matplotlib.collections.Collection):
        try:
            c.set_sizes(np.asarray(c.get_sizes()) * scale ** 2)
        except Exception:
            pass
    if not fig.get_constrained_layout():
        fig.tight_layout()
    # Raising small labels to the floor can make the drawing overrun the page box. Shrink the
    # canvas (not the type, which is already at the floor) until it fits.
    bw, bh = BOX[orient]
    for _ in range(8):
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        f = min(1.0, (bw - 0.1) / bb.width, (bh - 0.1) / bb.height)
        if f >= 1.0:
            break
        fig.set_size_inches(fig.get_size_inches() * f)
        if not fig.get_constrained_layout():
            fig.tight_layout()
    return orient


def save_supp(fig, letter, orient):
    fig.canvas.draw()
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    min_pt = min(t.get_fontsize() for t in texts)
    fams = sorted({t.get_fontname() for t in texts})
    over = text_overlaps(fig)
    leg = prf.legend_overlaps(fig)
    png = os.path.join(OUT, "Fig_%s.png" % letter)
    fig.savefig(png, dpi=prf.DPI, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)
    im = Image.open(png).convert("RGB")
    a = np.asarray(im)
    ys, xs = np.where((a < 250).any(axis=2))
    im = im.crop((max(0, xs.min() - 8), max(0, ys.min() - 8), min(im.width, xs.max() + 9), min(im.height, ys.max() + 9)))
    im.save(png, dpi=(prf.DPI, prf.DPI))
    w, h = im.width / prf.DPI, im.height / prf.DPI
    bw, bh = BOX[orient]
    REPORT[letter] = {"orientation": orient, "width_in": round(w, 2), "height_in": round(h, 2),
                      "min_font_pt": round(min_pt, 2), "fonts": fams, "text_overlaps": over, "legend_overlaps": leg}
    flag = ("  TEXT OVERLAP %s" % over[:3] if over else "") + ("  LEGEND %s" % leg[:3] if leg else "")
    print("Fig %s  %-9s %5.2f x %4.2f in  min %.1f pt  %s%s" % (letter, orient, w, h, min_pt, fams, flag))
    assert w <= bw + 0.05 and h <= bh + 0.05, (letter, w, h)
    assert min_pt >= prf.MIN_PT - 1e-6, (letter, min_pt)


def main():
    os.makedirs(OUT, exist_ok=True)
    logging.basicConfig(level=logging.WARNING)
    s13 = prf.load_module("src/reporting/step_13_figures_and_reports.py", "s13sup")
    data = s13.load_all_inputs(logging.getLogger("supp"))
    captured = {}
    s13.save_figure = lambda fig, output_dir, filename, **k: captured.__setitem__("fig", fig) or filename
    for letter, fn in FIGS:
        plt.rcParams.update(prf.STYLE)
        captured.clear()
        if letter == "A":
            fig = fig_A_breakdown(data)
        else:
            res = getattr(s13, fn)(data, OUT, logging.getLogger(fn))
            assert "fig" in captured, (fn, res)
            fig = captured["fig"]
        adjust(letter, fig)
        save_supp(fig, letter, normalise(fig))
    # external feature transfer: stored image, printed title cropped (see the module docstring)
    src = Image.open(os.path.join(PKG, "fig_external_feature_transfer.png")).convert("RGB")
    ink = (np.asarray(src) < 235).any(axis=2).any(axis=1)
    y = int(np.argmax(ink))
    while ink[y]:
        y += 1                                   # end of the title band
    im = src.crop((0, y, src.width, src.height))
    a = np.asarray(im)
    ys, xs = np.where((a < 250).any(axis=2))
    im = im.crop((max(0, xs.min() - 8), max(0, ys.min() - 8), xs.max() + 9, ys.max() + 9))
    im.save(os.path.join(OUT, "Fig_O.png"), dpi=(prf.DPI, prf.DPI))
    scale = min(1.0, BOX["landscape"][0] / (im.width / prf.DPI), BOX["landscape"][1] / (im.height / prf.DPI))
    REPORT["O"] = {"orientation": "landscape", "source": "stored image (step 18)",
                   "print_scale": round(scale, 2), "approx_print_pt_of_10pt_text": round(10 * scale, 1)}
    print("Fig O  landscape stored image, printed at scale %.2f (10-pt text -> %.1f pt)" % (scale, 10 * scale))
    json.dump(REPORT, open(os.path.join(OUT, "supp_figure_report.json"), "w", encoding="utf-8"), indent=2)
    bad = {k: v for k, v in REPORT.items() if v.get("text_overlaps") or v.get("legend_overlaps")}
    print("\n%d figures; with overlaps: %s" % (len(REPORT), ", ".join(sorted(bad)) or "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
