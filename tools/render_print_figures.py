# -*- coding: utf-8 -*-
"""Render the eight main-article figures at their printed size.

Why a separate renderer. The pipeline draws its figures for screen reading, 8-14 inches wide, in
DejaVu Sans. Scaled down to a 7.5-inch page width, their text printed at 4.5-6.7 pt, below the
usual journal minimum ("Arial, Times, or Symbol font only in 8-12 point"). Shrinking cannot fix
that, so every figure is drawn here directly at print size - width at most 7.5 in, 300 dpi, Arial,
no text below 8 pt - with no title inside the graphic (titles and legends belong in the caption).

What does not change. Every plotted value is read from the stored result files the pipeline wrote
(reports/tables, metadata) and, where the pipeline has a selection or calculation routine, that
routine is imported and called rather than rewritten: the representative-MRI selection
(step_13c), the power function (step_14), the step-05 result loader (step_13). No analysis is
re-run. `check_values` asserts the numbers each figure shows against the tables they come from.

    python tools/render_print_figures.py [--out DIR]
"""
import importlib.util
import io
import json
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
import pandas as pd
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.join(ROOT, "bcbm_project")
TABLES = os.path.join(PROJ, "reports", "tables")
META = os.path.join(PROJ, "metadata")
# output folder: --out DIR on the command line, results/figures/print_article by default
OUT = os.path.join(ROOT, "results", "figures", "print_article")
if "--out" in sys.argv:
    OUT = os.path.abspath(sys.argv[sys.argv.index("--out") + 1])
MAX_W_IN, MAX_H_IN, DPI, MIN_PT = 7.5, 8.75, 300, 8.0

STYLE = {
    "font.family": "Arial", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.title_fontsize": 8,
    "axes.linewidth": 0.8, "lines.linewidth": 1.4, "mathtext.fontset": "custom",
    "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "savefig.dpi": DPI, "pdf.fonttype": 42}
plt.rcParams.update(STYLE)

TCOL = {"target_er": "#E0735B", "target_pr": "#1B9E8A", "target_her2": "#2F3E55"}
TNAME = {"target_er": "ER", "target_pr": "PR", "target_her2": "HER2"}
TARGETS = ["target_er", "target_pr", "target_her2"]
REPORT = {}


def load_module(rel, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                    # dataclasses look their module up here
    spec.loader.exec_module(mod)
    plt.rcParams.update(STYLE)                 # the pipeline modules set their own style on import
    return mod


def panel_letter(ax, letter):
    ax.text(-0.13, 1.04, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def legend_overlaps(fig):
    """Anything a legend box covers: data markers, labels, or bars in the same axes. The first
    Fig 7 put its key over the HER2 row and hid that row's marker and p value; this catches it."""
    from matplotlib.legend import Legend
    r = fig.canvas.get_renderer()
    hits = []
    for ax in fig.axes:
        legs = [c for c in ax.get_children() if isinstance(c, Legend)]
        for lg in legs:
            bb = lg.get_window_extent(r).padded(2)
            for t in ax.texts:
                if t.get_visible() and t.get_text().strip() and bb.overlaps(t.get_window_extent(r)):
                    hits.append("text %r" % t.get_text())
            for ln in ax.lines:
                if ln.get_marker() in (None, "None", "", " "):
                    continue
                pts = ax.transData.transform(np.column_stack(ln.get_data()))
                if any(bb.contains(x, y) for x, y in pts):
                    hits.append("marker of %s" % (ln.get_label() or "series"))
            for p in ax.patches:
                if isinstance(p, patches.Rectangle) and p.get_height() > 0 and bb.overlaps(p.get_window_extent(r)):
                    hits.append("bar")
    return hits


def save(fig, n):
    """Save at print size, then check the printed text really is Arial and at least 8 pt."""
    fig.canvas.draw()
    hits = legend_overlaps(fig)
    assert not hits, ("Fig %d: a legend covers" % n, hits)
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    min_pt = min(t.get_fontsize() for t in texts)
    fams = sorted({t.get_fontname() for t in texts})
    png = os.path.join(OUT, "_Fig%d_preview.png" % n)
    fig.savefig(png, dpi=DPI, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)
    im = Image.open(png).convert("RGB")
    # "A 2-point white space border around each figure is recommended ... Crop out excess white
    # space around image content": crop to the ink and keep a 2-pt (about 8 px) border.
    a = np.asarray(im)
    ys, xs = np.where((a < 250).any(axis=2))
    pad = 8
    im = im.crop((max(0, xs.min() - pad), max(0, ys.min() - pad),
                  min(im.width, xs.max() + 1 + pad), min(im.height, ys.max() + 1 + pad)))
    im.save(png)
    w_in, h_in = im.width / DPI, im.height / DPI
    assert w_in <= MAX_W_IN + 0.01 and h_in <= MAX_H_IN + 0.01, ("Fig %d too large" % n, w_in, h_in)
    assert min_pt >= MIN_PT - 1e-6, ("Fig %d has text below 8 pt" % n, min_pt)
    assert fams == ["Arial"], ("Fig %d uses fonts other than Arial" % n, fams)
    tif = os.path.join(OUT, "Fig%d.tif" % n)
    im.save(tif, format="TIFF", compression="tiff_lzw", dpi=(DPI, DPI))
    REPORT["Fig%d" % n] = {"width_in": round(w_in, 2), "height_in": round(h_in, 2), "px": [im.width, im.height],
                            "min_font_pt": round(min_pt, 2), "fonts": fams,
                            "MB": round(os.path.getsize(tif) / 2 ** 20, 2)}
    print("Fig%d  %4.2f x %4.2f in  %4dx%-4d px  min font %.1f pt  %s" % (n, w_in, h_in, im.width, im.height, min_pt, fams))


# ---------------------------------------------------------------- Fig 1: cohort flow
def fig1(s13):
    lg = logging.getLogger("fig1")
    data = s13.load_all_inputs(lg)
    step03, step04 = data["step03"], data["step04"]
    n_les = len(step03)
    n_pat = s13.count_patients(step04)
    n_cases = s13.count_cases(step03)
    splits = step04.groupby("split")["patient_base"].nunique().to_dict()
    s3 = json.load(open(os.path.join(META, "step03_summary.json"), encoding="utf-8"))
    cats = s3["final_mask_category_counts"]
    total = int(sum(cats.values()))
    label = {"target": "target", "cavity_or_bed": "cavity/bed", "other_structure": "other structure",
             "manual_review_required": "manual review"}
    parts = {label.get(k, k): int(v) for k, v in cats.items() if k != "lesion"}
    resid = int(cats.get("lesion", 0)) - n_les
    if resid:
        parts["lesion, not eligible"] = resid
    excluded = total - n_les
    assert (total, n_les, n_pat, n_cases) == (2825, 1841, 139, 236), (total, n_les, n_pat, n_cases)
    assert (splits["train"], splits["valid"], splits["test"]) == (102, 20, 17), splits
    assert excluded == sum(parts.values()) == 984

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fills = ["#2E4057", "#048A81", "#E07A5F", "#F2CC8F"]
    ink = ["white", "white", "black", "black"]

    def box(x, y, w, h, txt, fc, tc="black", ec="black", ls="-", fs=9.5, bold=True):
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012", facecolor=fc,
                                            edgecolor=ec, linestyle=ls, linewidth=1.1))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color=tc, linespacing=1.35)

    ys = [0.855, 0.655, 0.455, 0.255]
    txts = ["Patients in the public collection\n%d" % 165, "Segmented structures\n%d" % total,
            "Analysis-eligible lesions\n%d" % n_les, "Patients: %d\nImaging cases: %d" % (n_pat, n_cases)]
    for y, t, fc, tc in zip(ys, txts, fills, ink):
        box(0.17, y, 0.46, 0.11, t, fc, tc)
    for y1, y2 in zip(ys[:-1], ys[1:]):
        ax.annotate("", xy=(0.40, y2 + 0.11 + 0.014), xytext=(0.40, y1 - 0.014),
                    arrowprops=dict(arrowstyle="-|>", lw=1.1, color="black"))
    ex_lines = "\n".join("%s = %d" % (k, v) for k, v in parts.items())
    box(0.70, 0.53, 0.28, 0.30, "Excluded structures: %d\n\n%s" % (excluded, ex_lines), "#EDEDED",
        tc="#222222", ec="#555555", ls="--", fs=8.5, bold=False)
    ax.annotate("", xy=(0.70 - 0.012, 0.71), xytext=(0.63 + 0.012, 0.71),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#555555", linestyle="--"))
    for i, (nm, key) in enumerate([("Training", "train"), ("Tuning", "valid"), ("Locked test", "test")]):
        box(0.05 + i * 0.26, 0.04, 0.20, 0.10, "%s\n%d patients" % (nm, splits[key]), "#F7F8FA")
        ax.annotate("", xy=(0.15 + i * 0.26, 0.14 + 0.013), xytext=(0.40, ys[3] - 0.014),
                    arrowprops=dict(arrowstyle="-|>", lw=1.0, color="black"))
    save(fig, 1)
    return {"total": total, "lesions": n_les, "patients": n_pat, "cases": n_cases, "excluded": excluded,
            "splits": [splits["train"], splits["valid"], splits["test"]]}


# ---------------------------------------------------------------- Fig 2: workflow schematic
def fig2():
    """The pipeline schematic is drawn by step_13b; it is re-drawn here with that code, in Arial,
    with its two title lines removed and every label raised to at least 8 pt at print size."""
    s13b = load_module("src/reporting/step_13b_methods_figure.py", "s13b")
    s13b.FONT = "Arial"
    captured = {}
    orig = matplotlib.figure.Figure.savefig

    def grab(self, *a, **k):
        captured["fig"] = self            # keep the figure, do not write the pipeline's file
    matplotlib.figure.Figure.savefig = grab
    close_orig = plt.close
    plt.close = lambda *a, **k: None
    try:
        s13b.main()
    finally:
        matplotlib.figure.Figure.savefig = orig
        plt.close = close_orig
    fig = captured["fig"]
    ax = fig.axes[0]
    for t in list(ax.texts):                   # the printed title and subtitle: removed, not hidden,
        s = t.get_text()                       # so the saved figure is cropped to the drawing
        if s.startswith("Leakage-Controlled Radiomics Pipeline") or s.startswith("Every label-dependent step"):
            t.remove()
    # The locked-test callout holds four lines in a box sized for smaller type; at the 8-pt floor
    # they touch its border. Give the box room downwards and space the lines evenly inside it.
    y0, h = 0.686, 0.078
    for p in ax.patches:
        if isinstance(p, patches.FancyBboxPatch) and abs(p.get_x() - 0.012) < 1e-6 and abs(p.get_y() - 0.700) < 1e-6:
            p.set_y(y0); p.set_height(h)
    lines = ["LOCKED TEST", "(17 patients)", "untouched until", "final evaluation"]
    for t in ax.texts:
        if t.get_text() in lines:
            i = lines.index(t.get_text())
            t.set_y(y0 + h - h * (i + 0.5) / 4)
    for c in ax.texts:                         # the dashed arrow leaves from the box's new bottom edge
        if isinstance(c, matplotlib.text.Annotation) and np.allclose(c.xyann, (0.095, 0.700)):
            c.xyann = (0.095, y0)
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    # 2% below the page limits: raising small labels to the floor makes the drawing a little taller
    scale = 0.98 * min(1.0, MAX_W_IN / bb.width, MAX_H_IN / bb.height)
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text().strip():
            t.set_fontfamily("Arial")
            t.set_fontsize(max(t.get_fontsize(), (MIN_PT + 0.2) / scale))
    fig.set_size_inches(fig.get_size_inches() * scale)
    for t in fig.findobj(Text):                # keep the same look at the new, smaller size
        t.set_fontsize(t.get_fontsize() * scale)
    for ln in fig.findobj(matplotlib.lines.Line2D):
        ln.set_linewidth(max(0.6, ln.get_linewidth() * scale))
    for p in fig.findobj(patches.Patch):
        p.set_linewidth(max(0.6, p.get_linewidth() * scale))
    save(fig, 2)


# ---------------------------------------------------------------- Fig 3: representative MRI
def fig3():
    s13c = load_module("src/reporting/step_13c_representative_mri.py", "s13c")
    lg = logging.getLogger("fig3")
    cand = s13c.load_candidates(lg)
    used, picks = set(), []
    for row_label, col, (neg_lab, pos_lab) in s13c.ROWS:
        masks = ([(cand["_fs"] == 1.5), (cand["_fs"] == 3.0)] if col == "_fs"
                 else [(cand[col] == 0), (cand[col] == 1)])
        picks.append((row_label, (neg_lab, s13c.pick_two(cand, masks[0], used, lg)),
                      (pos_lab, s13c.pick_two(cand, masks[1], used, lg))))
    fig, axes = plt.subplots(3, 4, figsize=(7.3, 5.9))
    vols = []
    for r, (row_label, (neg_lab, left), (pos_lab, right)) in enumerate(picks):
        for c in range(4):
            ax = axes[r][c]
            lab, color = (neg_lab, s13c.NEG_COLOR) if c < 2 else (pos_lab, s13c.POS_COLOR)
            group = left if c < 2 else right
            s13c.draw_panel(ax, group[c % 2], lab, color, lg)
            vols.append(round(float(group[c % 2]["cc"]), 1))
            for coll in ax.collections:        # thinner contour at the smaller panel size
                coll.set_linewidth(1.0)
            for t in ax.texts:
                if t.get_text().endswith("cc"):
                    # white-on-image volume labels vanished where the tissue is bright (the 3.7 read
                    # as 3.2, the 1.9 lost its 1): give them a dark backing so they read everywhere
                    t.set_fontsize(8.5)
                    t.set_bbox(dict(facecolor="black", alpha=0.6, edgecolor="none", pad=1.6))
                else:
                    t.set_fontsize(9)
        axes[r][0].set_ylabel(row_label, fontsize=9, fontweight="bold")
        axes[r][0].yaxis.set_visible(True); axes[r][0].set_yticks([])
    fig.tight_layout(rect=[0, 0, 1, 0.94], h_pad=0.4, w_pad=0.3)
    x_neg = (axes[0][0].get_position().x0 + axes[0][1].get_position().x1) / 2
    x_pos = (axes[0][2].get_position().x0 + axes[0][3].get_position().x1) / 2
    fig.text(x_neg, 0.955, "Negative", ha="center", fontsize=9.5, fontweight="bold", color=s13c.NEG_COLOR)
    fig.text(x_pos, 0.955, "Positive", ha="center", fontsize=9.5, fontweight="bold", color=s13c.POS_COLOR)
    save(fig, 3)
    return vols


# ---------------------------------------------------------------- Fig 4: generalization gap
def fig4(s13):
    data = s13.load_all_inputs(logging.getLogger("fig4"))
    d = s13.ensure_result_columns(data["step05_results"])
    fd = data.get("feature_dict") or {}
    n_rad = len(fd.get("radiomic_block_columns", [])) or 642
    n_acq = len(fd.get("acquisition_block_columns", [])) or 21
    n_bur = len(fd.get("burden_study_block_columns", [])) or 79
    mk = {"radiomics_pure": "o", "radiomics_plus_burden": "D", "all_features": "s", "acquisition_only": "^"}
    vl = {"radiomics_pure": "Radiomics only (%d)" % n_rad, "radiomics_plus_burden": "+ burden/study (%d)" % (n_rad + n_bur),
          "all_features": "All features (%d)" % (n_rad + n_bur + n_acq), "acquisition_only": "Acquisition only (%d)" % n_acq}
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.fill_between([0, 1], [0, 1], [0, 0], color="#FDEDEC", alpha=0.7, zorder=0)
    ax.fill_between([0, 1], [1, 1], [0, 1], color="#EAF7EA", alpha=0.7, zorder=0)
    pts = []
    for _, r in d.iterrows():
        x = s13.extract_metric(r, ["validation_AUROC_at_selection", "nested_cv_auroc_mean", "valid_AUROC"], np.nan)
        y = s13.extract_metric(r, ["AUROC"], np.nan)
        if np.isnan(x) or np.isnan(y):
            continue
        v = str(r.get("feature_set_variant", "radiomics_pure"))
        ax.scatter(x, y, s=46, color=TCOL[r["target"]], marker=mk.get(v, "o"), edgecolor="black",
                   linewidth=0.6, zorder=3)
        pts.append((r["target"], v, round(float(x), 3), round(float(y), 2)))
    ax.plot([0, 1], [0, 1], "--", color="black", lw=0.9, zorder=2)
    h1 = [Line2D([], [], marker="o", ls="", ms=6, markerfacecolor=TCOL[t], markeredgecolor="black", label=TNAME[t])
          for t in TARGETS]
    h2 = [Line2D([], [], marker=m, ls="", ms=6, markerfacecolor="white", markeredgecolor="black", label=vl[v])
          for v, m in mk.items()]
    l1 = ax.legend(handles=h1, title="Receptor", loc="upper left", framealpha=0.95)
    ax.add_artist(l1)
    ax.legend(handles=h2, title="Feature block", loc="lower right", framealpha=0.95)
    ax.text(0.62, 0.955, "locked test > nested CV", fontsize=8, color="#3F6B3F", style="italic")
    ax.text(0.80, 0.555, "locked test < nested CV", fontsize=8, color="#9A544C", style="italic", ha="center")
    ax.set_xlim(0.3, 0.9); ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("Nested cross-validation AUROC (single partition)")
    ax.set_ylabel("Locked-test AUROC")
    save(fig, 4)
    return pts


# ---------------------------------------------------------------- Fig 5: learning curve
def fig5():
    lc = pd.read_csv(os.path.join(TABLES, "step14_learning_curve.csv"))
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    ends, lo, hi = {}, 1.0, 0.0
    for t in TARGETS:
        s = lc[lc["target"] == t].sort_values("n_train_effective")
        x, m, sd = s["n_train_effective"].values, s["auroc_mean"].values, s["auroc_std"].values
        ax.plot(x, m, "-o", color=TCOL[t], lw=1.4, ms=3.5, label=TNAME[t])
        ax.fill_between(x, m - sd, m + sd, color=TCOL[t], alpha=0.14)
        ends[t] = [(int(a), round(float(b), 2)) for a, b in zip(x, m)]
        lo, hi = min(lo, float((m - sd).min())), max(hi, float((m + sd).max()))
    ax.axhline(0.5, color="gray", ls="--", lw=1.0, label="Chance (0.50)")
    ax.set_xlabel("Effective training-set size (patients)")
    ax.set_ylabel("Out-of-fold AUROC (mean ± SD)")
    # the whole ±1 SD band must be visible (the pipeline's 0.35 floor cut the PR band off)
    ax.set_ylim(np.floor((lo - 0.02) * 20) / 20, max(0.8, np.ceil((hi + 0.1) * 20) / 20))
    assert ax.get_ylim()[0] < lo and ax.get_ylim()[1] > hi
    ax.legend(loc="upper left", ncol=4, frameon=True, framealpha=0.95)
    ax.grid(alpha=0.25)
    save(fig, 5)
    return ends


# ---------------------------------------------------------------- Fig 6: power
def fig6():
    s14 = load_module("src/analysis/step_14_learning_curve_power.py", "s14")
    pw = pd.read_csv(os.path.join(TABLES, "step14_power_analysis.csv"))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={"width_ratios": [1, 1.08]})
    grid = np.linspace(0.5, 0.95, 181)
    lt = pw[pw["set"] == "locked_test"].set_index("target")
    op = pw[pw["set"] == "nested_oof_pool"].set_index("target")
    for t in TARGETS:
        r = lt.loc[t]
        y = [s14.power_auc(a, int(r["n_pos"]), int(r["n_neg"])) for a in grid]
        ax.plot(grid, y, color=TCOL[t], lw=1.4, label="%s (%d+/%d−)" % (TNAME[t], r["n_pos"], r["n_neg"]))
        ax.plot([r["min_detectable_auc_80pct"]], [0.8], "o", color=TCOL[t], ms=5, markeredgecolor="black", mew=0.6)
    ax.axhline(0.8, color="gray", ls="--", lw=1.0, label="80% power")
    ax.set_xlabel("True AUROC"); ax.set_ylabel("Power to detect AUROC > 0.50")
    ax.set_ylim(0, 1.02); ax.set_xlim(0.5, 0.95)
    ax.legend(loc="upper left", framealpha=0.95); ax.grid(alpha=0.25)
    panel_letter(ax, "A")
    x = np.arange(3); w = 0.27
    series = [([int(lt.loc[t, "n_total"]) for t in TARGETS], "#C0392B", "Locked test"),
              ([int(op.loc[t, "n_total"]) for t in TARGETS], "#2F3E55", "Nested-CV pool"),
              ([int(lt.loc[t, "required_total_n_auc070"]) for t in TARGETS], "#E8957F", "Needed for AUROC 0.70")]
    for k, (vals, col, lab) in enumerate(series):
        bars = ax2.bar(x + (k - 1) * w, vals, w, color=col, label=lab)
        for b in bars:
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, str(int(b.get_height())),
                     ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(x); ax2.set_xticklabels([TNAME[t] for t in TARGETS])
    ax2.set_ylabel("Evaluation-set size (patients)")
    ax2.set_ylim(0, 175)
    ax2.legend(loc="upper left", framealpha=0.95); ax2.grid(axis="y", alpha=0.25)
    panel_letter(ax2, "B")
    fig.tight_layout(w_pad=2.0)
    save(fig, 6)
    return {"test": series[0][0], "pool": series[1][0], "needed": series[2][0],
            "mda": [round(float(lt.loc[t, "min_detectable_auc_80pct"]), 3) for t in TARGETS]}


# ---------------------------------------------------------------- Fig 7: permutation null
def fig7():
    rb = pd.read_csv(os.path.join(TABLES, "step15_robustness.csv")).set_index("target")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.9))
    for i, t in enumerate(TARGETS):
        r = rb.loc[t]
        ax.barh(i, r["null_p95"] - 0.5, left=0.5, height=0.34, color=TCOL[t], alpha=0.28, zorder=1)
        ax.plot([r["null_mean"]], [i], "o", color=TCOL[t], ms=4.5, markeredgecolor="black", mew=0.6, zorder=3)
        ax.plot([r["observed_oof_auc"]], [i], "D", color=TCOL[t], ms=7, markeredgecolor="black", mew=0.6, zorder=4)
        ax.text(max(r["null_p95"], r["observed_oof_auc"]) + 0.018, i, "p = %.2f" % r["permutation_p_value"],
                va="center", fontsize=8)
    ax.axvline(0.5, color="gray", lw=0.9, ls=":")
    ax.set_yticks(range(3)); ax.set_yticklabels([TNAME[t] for t in TARGETS])
    # headroom above the HER2 row for the key, which otherwise covered HER2's marker and p value
    ax.set_xlabel("Out-of-fold AUROC"); ax.set_xlim(0.45, 0.72); ax.set_ylim(-0.6, 3.95)
    key = [patches.Patch(facecolor="#999999", alpha=0.35, label="Null: 0.50 to 95th percentile"),
           Line2D([], [], marker="o", ls="", ms=4.5, markerfacecolor="#999999", markeredgecolor="black", label="Null mean"),
           Line2D([], [], marker="D", ls="", ms=6, markerfacecolor="#999999", markeredgecolor="black", label="Observed")]
    ax.legend(handles=key, loc="upper right", framealpha=0.95, handlelength=1.2)
    panel_letter(ax, "A")
    for i, t in enumerate(TARGETS):
        r = rb.loc[t]
        ax2.errorbar(r["seed_auc_mean"], i, xerr=r["seed_auc_sd"], fmt="s", color=TCOL[t], ms=6,
                     markeredgecolor="black", mew=0.6, ecolor=TCOL[t], elinewidth=1.4, capsize=3, zorder=3)
        ax2.text(0.445, i + 0.22, "Brier %.2f" % r["brier"], fontsize=8, color="#333333")
    ax2.axvline(0.5, color="gray", lw=0.9, ls=":")
    ax2.set_yticks(range(3)); ax2.set_yticklabels([TNAME[t] for t in TARGETS])
    ax2.set_xlabel("Out-of-fold AUROC, mean ± SD over %d seeds" % int(rb["n_seeds"].iloc[0]))
    ax2.set_xlim(0.44, 0.62); ax2.set_ylim(-0.6, 3.95)       # rows aligned with panel A
    panel_letter(ax2, "B")
    fig.tight_layout(w_pad=2.0)
    save(fig, 7)
    return {TNAME[t]: {"p": round(float(rb.loc[t, "permutation_p_value"]), 2),
                       "seed_mean": round(float(rb.loc[t, "seed_auc_mean"]), 2),
                       "brier": round(float(rb.loc[t, "brier"]), 2)} for t in TARGETS}


# ---------------------------------------------------------------- Fig 8: optimism generalization
def fig8():
    res = pd.read_csv(os.path.join(TABLES, "step17_optimism_gap.csv"))
    gaps = res["optimism_gap_p95_minus_honest"].to_numpy()
    leaks = res["leakage_inflation"].dropna().to_numpy()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 2.9))
    a1.hist(gaps, bins=15, color="#E0735B", alpha=0.85, edgecolor="white")
    a1.axvline(float(np.median(gaps)), color="black", ls="--", lw=1.0, label="Median %.2f" % np.median(gaps))
    a1.set_xlabel("Single-split optimism gap\n(95th-percentile split − repeated CV AUROC)")
    a1.set_ylabel("Datasets"); a1.legend(loc="upper right")
    a1.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    panel_letter(a1, "A")
    a2.hist(leaks, bins=15, color="#2F3E55", alpha=0.85, edgecolor="white")
    a2.axvline(float(np.median(leaks)), color="black", ls="--", lw=1.0, label="Median %.2f" % np.median(leaks))
    a2.set_xlabel("Leakage inflation\n(selection on full data − repeated CV AUROC)")
    a2.set_ylabel("Datasets"); a2.legend(loc="upper left")
    a2.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    panel_letter(a2, "B")
    fig.tight_layout(w_pad=2.0)
    save(fig, 8)
    return {"n": len(res), "median_gap": round(float(np.median(gaps)), 2), "median_leak": round(float(np.median(leaks)), 2),
            "gap_range": [round(float(gaps.min()), 2), round(float(gaps.max()), 2)]}


def check_values(v):
    """The numbers each figure shows, against the reported tables."""
    assert v[1]["excluded"] == 984 and v[1]["splits"] == [102, 20, 17]
    # the lesion volumes the step-13c selection returns (3.77/3.71 cc for the two HER2-negative panels)
    assert v[3] == [1.9, 1.8, 2.6, 2.7, 3.8, 3.7, 1.7, 1.7, 2.1, 2.1, 1.9, 1.3], v[3]
    lt = {(t, var): y for t, var, _, y in v[4]}
    expect = {("target_er", "radiomics_pure"): 0.60, ("target_er", "radiomics_plus_burden"): 0.70,
              ("target_er", "all_features"): 0.52, ("target_er", "acquisition_only"): 0.57,
              ("target_pr", "radiomics_pure"): 0.62, ("target_pr", "radiomics_plus_burden"): 0.62,
              ("target_pr", "all_features"): 0.62, ("target_pr", "acquisition_only"): 0.50,
              ("target_her2", "radiomics_pure"): 0.50, ("target_her2", "radiomics_plus_burden"): 0.56,
              ("target_her2", "all_features"): 0.42, ("target_her2", "acquisition_only"): 0.78}
    assert lt == expect, "Fig 4 points differ from the locked-test column of Table 2"
    e = v[5]
    assert e["target_er"][0] == (29, 0.50) and e["target_er"][-1] == (95, 0.50), e["target_er"]
    assert e["target_her2"][0][1] == 0.52 and e["target_her2"][-1] == (92, 0.56), e["target_her2"]
    assert [p[1] for p in e["target_pr"][-2:]] == [0.41, 0.43], e["target_pr"]
    assert v[6]["test"] == [16, 16, 17] and v[6]["pool"] == [119, 116, 115] and v[6]["needed"] == [63, 72, 63]
    assert v[7] == {"ER": {"p": 0.12, "seed_mean": 0.50, "brier": 0.26}, "PR": {"p": 0.02, "seed_mean": 0.58, "brier": 0.24},
                    "HER2": {"p": 0.07, "seed_mean": 0.50, "brier": 0.25}}, v[7]
    assert v[8]["n"] == 18 and v[8]["median_gap"] == 0.17 and v[8]["median_leak"] == 0.08 and v[8]["gap_range"] == [0.07, 0.32], v[8]
    print("values checked against the reported tables: Figs 1, 3, 4, 5, 6, 7, 8")


def main():
    os.makedirs(OUT, exist_ok=True)
    logging.basicConfig(level=logging.WARNING)
    s13 = load_module("src/reporting/step_13_figures_and_reports.py", "s13")
    v = {1: fig1(s13)}
    fig2()
    v[3] = fig3()
    v[4] = fig4(s13)
    v[5] = fig5()
    v[6] = fig6()
    v[7] = fig7()
    v[8] = fig8()
    check_values(v)
    json.dump(REPORT, open(os.path.join(OUT, "figure_report.json"), "w", encoding="utf-8"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
