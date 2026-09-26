"""
Step 16 — Radiomics preprocessing-sensitivity analysis
======================================================
Quantifies how much nested-CV AUROC and feature values shift when the radiomics
*extraction* settings change — demonstrating the fragility that underlies the
non-reproducibility / non-transfer of radiomics. CPU-only, but slow (re-extracts
features over all lesions for several settings).

Settings varied (PyRadiomics):
  - binWidth: 10, 25, 50
  - resampledPixelSpacing: none vs [1,1,1] mm (BraTS/IBSI-style isotropic)
  - normalize: False vs True (z-score intensity normalization)

For each setting it re-extracts firstorder+shape+GLCM features per lesion,
aggregates to the patient level (median), and computes the light leakage-safe
nested-CV AUROC (same reduced pipeline as Step 15) per target. The spread of
AUROC across settings is the sensitivity.

Requires: pyradiomics, nibabel/SimpleITK (the same stack as Step 03).

Usage:
  python src/analysis/step_16_preprocessing_sensitivity.py [--max-lesions N]

Outputs:
  reports/figures/step13_publication_package/fig_preprocessing_sensitivity.{png,pdf}
  reports/tables/step16_preprocessing_sensitivity.csv
  metadata/step16_preprocessing_sensitivity_summary.json
"""
from __future__ import annotations

import argparse, json, logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

_REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = _REPO_ROOT / "bcbm_project"
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = Path("./bcbm_project").resolve()
STEP_NAME = "step_16_preprocessing_sensitivity"
LESION_CSV = PROJECT_ROOT / "data" / "processed" / "analysis_ready_step03_lesion_only.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "step13_publication_package"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
METADATA_DIR = PROJECT_ROOT / "metadata"
LOGS_DIR = PROJECT_ROOT / "logs"

TARGETS = {"ER_bin": "ER", "PR_bin": "PR", "HER2_bin": "HER2"}
TARGET_COLORS = {"ER": "#E0735B", "PR": "#1B9E8A", "HER2": "#2F3E55"}
GLOBAL_SEED = 42
N_SPLITS = 5

SETTINGS = [
    {"name": "binWidth=25, native",      "binWidth": 25, "resample": None,      "normalize": False},
    {"name": "binWidth=10, native",      "binWidth": 10, "resample": None,      "normalize": False},
    {"name": "binWidth=50, native",      "binWidth": 50, "resample": None,      "normalize": False},
    {"name": "binWidth=25, 1mm iso",     "binWidth": 25, "resample": [1, 1, 1], "normalize": False},
    {"name": "binWidth=25, z-norm",      "binWidth": 25, "resample": None,      "normalize": True},
]


def setup_logger():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(STEP_NAME); lg.setLevel(logging.INFO)
    if not lg.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        sh = logging.StreamHandler(); sh.setFormatter(fmt); lg.addHandler(sh)
        fh = logging.FileHandler(LOGS_DIR / f"{STEP_NAME}.log", encoding="utf-8"); fh.setFormatter(fmt); lg.addHandler(fh)
    return lg


def extract_features(df, setting, logger, max_lesions: Optional[int]):
    from radiomics import featureextractor
    params = {"binWidth": setting["binWidth"], "label": 1, "normalize": setting["normalize"]}
    if setting["resample"] is not None:
        params["resampledPixelSpacing"] = setting["resample"]
        params["interpolator"] = "sitkBSpline"
    ex = featureextractor.RadiomicsFeatureExtractor(**params)
    ex.disableAllFeatures()
    for cls in ("firstorder", "shape", "glcm"):
        ex.enableFeatureClassByName(cls)
    rows = []
    sub = df if max_lesions is None else df.head(max_lesions)
    for i, (_, r) in enumerate(sub.iterrows()):
        try:
            res = ex.execute(str(r["image_abs_path"]), str(r["mask_abs_path"]))
            feat = {k: float(v) for k, v in res.items() if not k.startswith("diagnostics_")}
            feat["patient_base"] = r["patient_base"]
            for t in TARGETS:
                feat[t] = r.get(t)
            rows.append(feat)
        except Exception as e:
            logger.debug("lesion %d failed: %s", i, e)
        if (i + 1) % 200 == 0:
            logger.info("    extracted %d/%d", i + 1, len(sub))
    return pd.DataFrame(rows)


def light_oof_auc(X, y, seed=GLOBAL_SEED):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    for tr, va in skf.split(X, y):
        try:
            p = Pipeline([("imp", SimpleImputer(strategy="median")), ("var", VarianceThreshold(1e-8)),
                          ("sc", RobustScaler()),
                          ("m", LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.5,
                                                   class_weight="balanced", max_iter=4000, random_state=seed))])
            p.fit(X[tr], y[tr]); oof[va] = p.predict_proba(X[va])[:, 1]
        except Exception:
            pass
    m = ~np.isnan(oof)
    return float(roc_auc_score(y[m], oof[m])) if m.sum() and len(np.unique(y[m])) > 1 else np.nan


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--max-lesions", type=int, default=None); a = ap.parse_args()
    for d in (FIGURES_DIR, TABLES_DIR, METADATA_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(); logger.info("=" * 80); logger.info("Starting %s", STEP_NAME)
    try:
        import radiomics  # noqa
    except Exception:
        logger.error("pyradiomics not installed. Run: pip install pyradiomics"); raise SystemExit("pyradiomics missing")
    if not LESION_CSV.exists():
        raise FileNotFoundError(f"Run steps 00–03 first; missing {LESION_CSV}")

    df = pd.read_csv(LESION_CSV)
    df = df[df["final_mask_category"].astype(str).str.lower() == "lesion"]
    df = df[df["image_abs_path"].apply(lambda p: isinstance(p, str) and Path(p).exists())]
    df = df[df["mask_abs_path"].apply(lambda p: isinstance(p, str) and Path(p).exists())]
    logger.info("lesions: %d", len(df))

    rows: List[Dict] = []
    for setting in SETTINGS:
        logger.info("  setting: %s", setting["name"])
        feat = extract_features(df, setting, logger, a.max_lesions)
        if feat.empty:
            continue
        fcols = [c for c in feat.columns if c.startswith("original_")]
        # aggregate lesion -> patient (median)
        agg = feat.groupby("patient_base").agg({**{c: "median" for c in fcols},
                                                **{t: "first" for t in TARGETS}}).reset_index()
        for tcol, tname in TARGETS.items():
            d = agg[agg[tcol].notna()].copy()
            y = d[tcol].astype(int).to_numpy()
            X = d[fcols].apply(pd.to_numeric, errors="coerce").to_numpy()
            if len(np.unique(y)) < 2:
                continue
            auc = light_oof_auc(X, y)
            rows.append({"setting": setting["name"], "target": tname, "auc": round(auc, 4) if not np.isnan(auc) else None,
                         "n_patients": int(len(d)), "n_features": len(fcols)})
            logger.info("    %s | %s AUROC=%.3f", setting["name"], tname, auc)

    if not rows:
        logger.error("no results"); raise SystemExit(1)
    res = pd.DataFrame(rows)
    res.to_csv(TABLES_DIR / "step16_preprocessing_sensitivity.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    settings = [s["name"] for s in SETTINGS]
    x = np.arange(len(settings)); w = 0.25
    for j, tname in enumerate(["ER", "PR", "HER2"]):
        vals = [res[(res.setting == s) & (res.target == tname)]["auc"].max() if not res[(res.setting == s) & (res.target == tname)].empty else np.nan for s in settings]
        ax.bar(x + (j - 1) * w, vals, w, color=TARGET_COLORS[tname], label=tname)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(settings, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Out-of-fold AUROC"); ax.set_ylim(0.35, 0.75)
    ax.set_title("Radiomics preprocessing sensitivity (extraction settings)", fontweight="bold")
    ax.legend(); fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_preprocessing_sensitivity.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig_preprocessing_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)

    spread = res.groupby("target")["auc"].agg(["min", "max"]).reset_index()
    spread["range"] = spread["max"] - spread["min"]
    summary = {"step": STEP_NAME, "settings": [s["name"] for s in SETTINGS],
               "auc_spread_by_target": spread.to_dict("records"), "results": rows}
    (METADATA_DIR / "step16_preprocessing_sensitivity_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Completed %s.", STEP_NAME); logger.info("=" * 80)


if __name__ == "__main__":
    main()
