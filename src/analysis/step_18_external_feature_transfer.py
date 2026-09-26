"""
Step 18 — External feature-transfer / domain-shift analysis (BCBM vs UCSF-BMSR)
==============================================================================
Tests whether BCBM-derived radiomic features TRANSFER to an independent public
breast-brain-metastasis cohort (the breast-primary subset of UCSF-BMSR). This is
a *feature-transfer / domain-shift* analysis, NOT external receptor validation:
no public brain-metastasis dataset provides ER/PR/HER2 labels, so no external
AUROC for receptor status can be computed. It quantifies whether the radiomic
feature space itself generalizes across institutions/scanners.

CPU-only. LOCAL-ONLY: never transmits image data anywhere; reads local NIfTI and
writes aggregate cohort-level statistics/figures only (no patient identifiers).
Complies with the UCSF-BMSR Data Use Agreement (research/non-commercial, no
re-identification, no transfer outside the licensee organization). Cite UCSF-BMSR
(Rudie et al., Radiol Artif Intell 2024) in any resulting publication.

Parity with the BCBM pipeline
  - BCBM: per-lesion masks -> extract per lesion -> aggregate to patient (median).
  - UCSF: binary metastasis seg -> connected-component labelling (one label per
    metastasis) -> extract per component -> aggregate to exam (median).
  - Identical PyRadiomics settings for BOTH cohorts (IBSI classes, 1-mm isotropic
    resampling, bin width 25) so features are comparable.

Verified against the actual UCSF-BMSR TRAIN download:
  external_ucsf_bmsr/
    TableS1_UCSF_BrainMetastases_SubjectInfo.xlsx   (sheet RENAMED; 461 exams)
      columns incl. SubjectID, CancerType (Breast=113), Scanner Strength (Tesla), ...
    UCSF_BrainMetastases_TRAIN/<SubjectID>/<SubjectID>_T1post.nii.gz
                                          /<SubjectID>_seg.nii.gz   (binary 0/1)

Run in the dedicated env:
    conda activate bcbm_radiomics
    python src/analysis/step_18_external_feature_transfer.py [--dry-run] [--max-lesions N]

Outputs:
  reports/figures/step13_publication_package/fig_external_feature_transfer.{png,pdf}
  reports/tables/step18_external_feature_transfer.csv
  metadata/step18_external_feature_transfer_summary.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy import stats

# ----------------------------------------------------------------------------- paths
_REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = _REPO_ROOT / "bcbm_project"
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = Path("./bcbm_project").resolve()
STEP_NAME = "step_18_external_feature_transfer"
LESION_CSV = PROJECT_ROOT / "data" / "processed" / "analysis_ready_step03_lesion_only.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "step13_publication_package"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
METADATA_DIR = PROJECT_ROOT / "metadata"
LOGS_DIR = PROJECT_ROOT / "logs"
DEFAULT_EXTERNAL_ROOT = PROJECT_ROOT / "data" / "external_ucsf_bmsr"

GLOBAL_SEED = 42
TARGETS = {"ER_bin": "ER", "PR_bin": "PR", "HER2_bin": "HER2"}

# ---- extraction settings (identical for both cohorts) ----
# Fixed bin COUNT (not width): the raw UCSF and BCBM intensity scales differ by orders
# of magnitude, so a fixed binWidth yields thousands of gray levels on some UCSF lesions
# and blows up the GLCM matrix (MemoryError). A fixed bin count bounds the gray levels,
# is IBSI-permitted, and gives a consistent, scanner-scale-robust discretization for BOTH
# cohorts. Linear interpolator (BSpline is prohibitively slow on the large UCSF volumes).
EXTRACT = {"binCount": 32, "label": 1, "normalize": False,
           "resampledPixelSpacing": [1, 1, 1], "interpolator": "sitkLinear"}
CROP_PAD = 10                     # voxels of margin around each lesion bounding box
FEATURE_CLASSES = ("firstorder", "shape", "glcm", "glrlm", "glszm", "gldm", "ngtdm")
MIN_COMPONENT_VOXELS = 8          # skip sub-voxel/noise components
MAX_NONBREAST_EXAMS = 90          # cap non-breast controls to bound compute

# ---- UCSF-BMSR layout (verified) ----
EXT_TRAIN_SUBDIR = "UCSF_BrainMetastases_TRAIN"
EXT_ID_COL = "SubjectID"
EXT_PRIMARY_COL = "CancerType"
EXT_BREAST_REGEX = r"breast"
EXT_IMAGE_SUFFIX = "_T1post.nii.gz"
EXT_MASK_SUFFIX = "_seg.nii.gz"


def setup_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(STEP_NAME); lg.setLevel(logging.INFO)
    if not lg.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        sh = logging.StreamHandler(); sh.setFormatter(fmt); lg.addHandler(sh)
        fh = logging.FileHandler(LOGS_DIR / f"{STEP_NAME}.log", encoding="utf-8"); fh.setFormatter(fmt); lg.addHandler(fh)
    return lg


# ---------------------------------------------------------------- external loader
def read_external_metadata(external_root: Path, logger) -> pd.DataFrame:
    meta = None
    for pat in ("*.xlsx", "*.xls", "*.csv"):
        for f in external_root.glob(pat):
            try:
                meta = pd.read_excel(f) if f.suffix.lower().startswith(".xls") else pd.read_csv(f)
                logger.info("metadata: %s (%d rows, %d cols)", f.name, len(meta), meta.shape[1])
                break
            except Exception:
                continue
        if meta is not None:
            break
    if meta is None:
        raise RuntimeError("No metadata table found under external root.")
    if EXT_ID_COL not in meta.columns or EXT_PRIMARY_COL not in meta.columns:
        raise RuntimeError(f"Expected columns '{EXT_ID_COL}' and '{EXT_PRIMARY_COL}'; "
                           f"found {list(meta.columns)}")
    return meta


def locate_external_cases(external_root: Path, logger) -> List[Dict]:
    meta = read_external_metadata(external_root, logger)
    tdir = external_root / EXT_TRAIN_SUBDIR
    if not tdir.exists():
        tdir = external_root  # fall back: cases directly under root
    is_breast = meta[EXT_PRIMARY_COL].astype(str).str.contains(EXT_BREAST_REGEX, case=False, na=False)
    logger.info("metadata primaries: breast=%d, non-breast=%d", int(is_breast.sum()), int((~is_breast).sum()))

    def build(rows, breast_flag, cap=None):
        out = []
        for _, r in rows.iterrows():
            sid = str(r[EXT_ID_COL]).strip()
            cdir = tdir / sid
            img, msk = cdir / f"{sid}{EXT_IMAGE_SUFFIX}", cdir / f"{sid}{EXT_MASK_SUFFIX}"
            if img.exists() and msk.exists():
                out.append({"case_id": sid, "image_path": str(img), "mask_path": str(msk),
                            "is_breast": breast_flag, "primary": str(r[EXT_PRIMARY_COL])})
            if cap and len(out) >= cap:
                break
        return out

    breast_cases = build(meta[is_breast], 1)
    control_cases = build(meta[~is_breast], 0, cap=MAX_NONBREAST_EXAMS)
    logger.info("on-disk exams paired: breast=%d, non-breast(control, capped)=%d",
                len(breast_cases), len(control_cases))
    return breast_cases + control_cases


# ---------------------------------------------------------------- extraction
def _make_extractor():
    from radiomics import featureextractor
    ex = featureextractor.RadiomicsFeatureExtractor(**EXTRACT)
    ex.disableAllFeatures()
    for cls in FEATURE_CLASSES:
        ex.enableFeatureClassByName(cls)
    return ex


def extract_external_lesions(cases: List[Dict], logger, max_lesions: Optional[int]) -> pd.DataFrame:
    """Connected-component (per-metastasis) extraction to mirror BCBM per-lesion features."""
    import SimpleITK as sitk
    ex = _make_extractor()
    rows: List[Dict] = []
    n_lesions = 0
    for i, c in enumerate(cases):
        try:
            img = sitk.ReadImage(c["image_path"])
            m = sitk.ReadImage(c["mask_path"])
            if m.GetSize() != img.GetSize():      # rare geometry mismatch -> mask to image grid
                m = sitk.Resample(m, img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, m.GetPixelID())
            binm = sitk.BinaryThreshold(m, 1, 1e9, 1, 0)
            cc = sitk.ConnectedComponent(binm)
            cc = sitk.RelabelComponent(cc, minimumObjectSize=MIN_COMPONENT_VOXELS)
            stats_f = sitk.LabelShapeStatisticsImageFilter(); stats_f.Execute(cc)
            dim = img.GetSize()
            for lab in stats_f.GetLabels():
                try:
                    # crop to the lesion bounding box (+pad) BEFORE extraction so resampling
                    # operates on a small ROI, not the full high-res volume.
                    bb = stats_f.GetBoundingBox(int(lab))  # (x,y,z,sx,sy,sz)
                    idx = [max(0, bb[k] - CROP_PAD) for k in range(3)]
                    size = [min(dim[k] - idx[k], bb[3 + k] + 2 * CROP_PAD) for k in range(3)]
                    img_c = sitk.RegionOfInterest(img, size, idx)
                    cc_c = sitk.RegionOfInterest(cc, size, idx)
                    res = ex.execute(img_c, cc_c, label=int(lab))
                    feat = {k: float(v) for k, v in res.items() if not k.startswith("diagnostics_")}
                    feat["case_id"] = c["case_id"]; feat["is_breast"] = c["is_breast"]
                    rows.append(feat); n_lesions += 1
                except Exception as e:
                    logger.debug("  %s label %s failed: %s", c["case_id"], lab, e)
                if max_lesions and n_lesions >= max_lesions:
                    break
        except Exception as e:
            logger.debug("case %s failed: %s", c["case_id"], e)
        if (i + 1) % 20 == 0:
            logger.info("    external exams processed %d/%d (%d lesions)", i + 1, len(cases), n_lesions)
        if max_lesions and n_lesions >= max_lesions:
            break
    logger.info("external lesions extracted: %d", len(rows))
    return pd.DataFrame(rows)


def extract_bcbm(logger, max_lesions: Optional[int]) -> pd.DataFrame:
    """Per-lesion BCBM extraction with identical settings (label=1 per lesion mask).
    Crops to the lesion bounding box first (as for the external cohort) so resampling
    is cheap and the run is fast."""
    import SimpleITK as sitk
    ex = _make_extractor()
    df = pd.read_csv(LESION_CSV)
    df = df[df["final_mask_category"].astype(str).str.lower() == "lesion"]
    df = df[df["image_abs_path"].apply(lambda p: isinstance(p, str) and Path(p).exists())]
    df = df[df["mask_abs_path"].apply(lambda p: isinstance(p, str) and Path(p).exists())]
    logger.info("BCBM internal lesions to extract: %d", len(df))
    rows = []
    sub = df if max_lesions is None else df.head(max_lesions)
    for i, (_, r) in enumerate(sub.iterrows()):
        try:
            img = sitk.ReadImage(str(r["image_abs_path"]))
            m = sitk.ReadImage(str(r["mask_abs_path"]))
            if m.GetSize() != img.GetSize():
                m = sitk.Resample(m, img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, m.GetPixelID())
            binm = sitk.BinaryThreshold(m, 1, 1e9, 1, 0)
            st = sitk.LabelShapeStatisticsImageFilter(); st.Execute(binm)
            if 1 not in st.GetLabels():
                continue
            bb = st.GetBoundingBox(1); dim = img.GetSize()
            idx = [max(0, bb[k] - CROP_PAD) for k in range(3)]
            size = [min(dim[k] - idx[k], bb[3 + k] + 2 * CROP_PAD) for k in range(3)]
            res = ex.execute(sitk.RegionOfInterest(img, size, idx),
                             sitk.RegionOfInterest(binm, size, idx), label=1)
            feat = {k: float(v) for k, v in res.items() if not k.startswith("diagnostics_")}
            feat["case_id"] = r["patient_base"]; feat["is_breast"] = 1
            rows.append(feat)
        except Exception as e:
            logger.debug("BCBM lesion %d failed: %s", i, e)
        if (i + 1) % 300 == 0:
            logger.info("    BCBM extracted %d/%d", i + 1, len(sub))
    return pd.DataFrame(rows)


def to_patient_level(feat: pd.DataFrame):
    fcols = [c for c in feat.columns if c.startswith("original_")]
    agg = feat.groupby("case_id").agg({**{c: "median" for c in fcols},
                                       "is_breast": "first"}).reset_index()
    return agg, fcols


# ---------------------------------------------------------------- analyses
def domain_shift(bcbm: np.ndarray, ext: np.ndarray, fcols: List[str]) -> pd.DataFrame:
    rows = []
    for j, f in enumerate(fcols):
        a, b = bcbm[:, j], ext[:, j]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 3 or len(b) < 3:
            continue
        ks, p = stats.ks_2samp(a, b)
        sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) or 1.0
        rows.append({"feature": f, "ks": round(float(ks), 4), "p": float(p),
                     "smd": round(float((a.mean() - b.mean()) / sd), 4)})
    res = pd.DataFrame(rows)
    if len(res):
        order = res["p"].rank(method="first"); m = len(res)
        res["p_fdr"] = (res["p"] * m / order).clip(upper=1.0)
    return res


def combat_lite(bcbm: np.ndarray, ext: np.ndarray) -> np.ndarray:
    """Location-scale align external features to the BCBM mean/scale (per feature)."""
    mu_b, sd_b = np.nanmean(bcbm, 0), np.nanstd(bcbm, 0) + 1e-8
    mu_e, sd_e = np.nanmean(ext, 0), np.nanstd(ext, 0) + 1e-8
    return (ext - mu_e) / sd_e * sd_b + mu_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--external-root", type=str, default=str(DEFAULT_EXTERNAL_ROOT))
    ap.add_argument("--max-lesions", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="Force re-extraction (ignore feature cache).")
    a = ap.parse_args()
    for d in (FIGURES_DIR, TABLES_DIR, METADATA_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(); logger.info("=" * 80); logger.info("Starting %s", STEP_NAME)

    ext_root = Path(a.external_root)
    cases = locate_external_cases(ext_root, logger)
    n_breast = sum(c["is_breast"] for c in cases)
    logger.info("located %d exams (%d breast) under %s", len(cases), n_breast, ext_root)
    if a.dry_run:
        for c in cases[:8]:
            logger.info("  %s | breast=%d | %s + %s", c["case_id"], c["is_breast"],
                        Path(c["image_path"]).name, Path(c["mask_path"]).name)
        logger.info("DRY-RUN complete (%d breast, %d control). Re-run without --dry-run to extract.",
                    n_breast, len(cases) - n_breast)
        return
    if n_breast < 5:
        raise SystemExit(f"Only {n_breast} breast exams paired; check layout/CONFIG.")
    try:
        import radiomics  # noqa
    except Exception:
        logger.error("pyradiomics not installed. Use: conda activate bcbm_radiomics"); raise SystemExit("pyradiomics missing")

    cache_b = METADATA_DIR / "step18_cache_bcbm_features.csv"
    cache_e = METADATA_DIR / "step18_cache_external_features.csv"
    if not a.refresh and cache_b.exists() and cache_e.exists() and a.max_lesions is None:
        logger.info("Loading cached patient-level features (skip extraction; use --refresh to re-extract).")
        bcbm_p = pd.read_csv(cache_b); ext_p = pd.read_csv(cache_e)
    else:
        logger.info("Extracting external lesions (connected components) ...")
        ext_feat = extract_external_lesions(cases, logger, a.max_lesions)
        logger.info("Extracting BCBM lesions (identical settings) ...")
        bcbm_feat = extract_bcbm(logger, a.max_lesions)
        if ext_feat.empty or bcbm_feat.empty:
            raise SystemExit("Extraction produced no features; aborting.")
        ext_p, _ = to_patient_level(ext_feat)
        bcbm_p, _ = to_patient_level(bcbm_feat)
        if a.max_lesions is None:
            bcbm_p.to_csv(cache_b, index=False); ext_p.to_csv(cache_e, index=False)
    ext_cols = [c for c in ext_p.columns if c.startswith("original_")]
    bcbm_cols = [c for c in bcbm_p.columns if c.startswith("original_")]
    fcols = sorted(set(ext_cols) & set(bcbm_cols))
    ext_breast = ext_p[ext_p["is_breast"] == 1]
    logger.info("common features=%d | BCBM exams=%d | external breast exams=%d | control=%d",
                len(fcols), len(bcbm_p), len(ext_breast), int((ext_p["is_breast"] == 0).sum()))

    Xb = bcbm_p[fcols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    Xe = ext_breast[fcols].apply(pd.to_numeric, errors="coerce").to_numpy(float)

    shift = domain_shift(Xb, Xe, fcols)
    frac_shift = float((shift["p_fdr"] < 0.05).mean()) if len(shift) else float("nan")
    median_smd = float(shift["smd"].abs().median()) if len(shift) else float("nan")
    shift.sort_values("p").to_csv(TABLES_DIR / "step18_external_feature_transfer.csv",
                                  index=False, encoding="utf-8-sig")

    imp = SimpleImputer(strategy="median").fit(Xb)
    Xb_i, Xe_i = imp.transform(Xb), imp.transform(Xe)
    Xe_aligned = combat_lite(Xb_i, Xe_i)
    shift_after = domain_shift(Xb_i, Xe_aligned, fcols)
    frac_shift_after = float((shift_after["p_fdr"] < 0.05).mean()) if len(shift_after) else float("nan")

    # Robust scaling on the pooled cohorts + percentile axis limits so a few extreme
    # out-of-distribution external exams do not collapse the overlay to the origin.
    Xpool = np.vstack([Xb_i, Xe_i])
    sc = RobustScaler().fit(Xpool)
    _z = lambda X: np.clip(sc.transform(X), -6, 6)   # winsorize extreme z-scores
    pca = PCA(n_components=2, random_state=GLOBAL_SEED).fit(_z(Xpool))
    Pb, Pe = pca.transform(_z(Xb_i)), pca.transform(_z(Xe_i))

    # model transfer (no labels): elastic-net trained on BCBM radiomics -> external scores
    score_stats = {}
    labels = _load_bcbm_labels(bcbm_p["case_id"].tolist())
    for tcol, tname in TARGETS.items():
        y = labels.get(tcol)
        if y is None:
            continue
        mask = ~np.isnan(y)
        if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
            continue
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("var", VarianceThreshold(1e-8)),
                         ("sc", RobustScaler()),
                         ("m", LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5,
                                                  C=0.5, class_weight="balanced", max_iter=4000,
                                                  random_state=GLOBAL_SEED))])
        try:
            pipe.fit(Xb[mask], y[mask].astype(int))
            s = pipe.predict_proba(Xe)[:, 1]
            score_stats[tname] = {"mean": round(float(s.mean()), 4), "sd": round(float(s.std()), 4),
                                  "min": round(float(s.min()), 4), "max": round(float(s.max()), 4)}
        except Exception as e:
            logger.debug("model transfer %s failed: %s", tname, e)

    # positive control: breast vs other on external
    pos_ctrl = None
    if (ext_p["is_breast"] == 0).sum() >= 10:
        Xall = ext_p[fcols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        yall = ext_p["is_breast"].to_numpy(int)
        oof = np.full(len(yall), np.nan)
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=GLOBAL_SEED).split(Xall, yall):
            p = Pipeline([("imp", SimpleImputer(strategy="median")), ("var", VarianceThreshold(1e-8)),
                          ("sc", RobustScaler()),
                          ("m", LogisticRegression(max_iter=4000, class_weight="balanced",
                                                   random_state=GLOBAL_SEED))])
            try:
                p.fit(Xall[tr], yall[tr]); oof[va] = p.predict_proba(Xall[va])[:, 1]
            except Exception:
                pass
        mm = ~np.isnan(oof)
        if mm.sum() and len(np.unique(yall[mm])) > 1:
            pos_ctrl = round(float(roc_auc_score(yall[mm], oof[mm])), 4)

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(Pb[:, 0], Pb[:, 1], s=16, alpha=0.55, label=f"BCBM (n={len(Pb)})", color="#2F3E55")
    ax[0].scatter(Pe[:, 0], Pe[:, 1], s=30, alpha=0.75, marker="^",
                  label=f"UCSF-BMSR breast (n={len(Pe)})", color="#E0735B")
    ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2"); ax[0].legend()
    _allP = np.vstack([Pb, Pe]); _lo, _hi = np.percentile(_allP, [2, 98], axis=0)
    _pad = 0.12 * (_hi - _lo) + 1e-6
    ax[0].set_xlim(_lo[0] - _pad[0], _hi[0] + _pad[0]); ax[0].set_ylim(_lo[1] - _pad[1], _hi[1] + _pad[1])
    ax[0].set_title("A. Radiomic feature space (PCA overlay)", fontweight="bold", fontsize=11)
    ax[1].bar(["raw", "after location-scale\nharmonization"], [frac_shift * 100, frac_shift_after * 100],
              color=["#E0735B", "#1B9E8A"], width=0.5)
    ax[1].set_ylabel("% of 107 features shifted (KS, FDR<0.05)"); ax[1].set_ylim(0, 100)
    for k, v in enumerate([frac_shift * 100, frac_shift_after * 100]):
        ax[1].text(k, v + 2, f"{v:.0f}%", ha="center", fontweight="bold")
    ax[1].set_title("B. Cross-cohort feature shift", fontweight="bold", fontsize=11)
    fig.suptitle("External feature transfer: BCBM → UCSF-BMSR breast subset "
                 "(domain shift; no receptor labels)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIGURES_DIR / "fig_external_feature_transfer.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig_external_feature_transfer.pdf", bbox_inches="tight")
    plt.close(fig)

    summary = {"step": STEP_NAME, "external_dataset": "UCSF-BMSR (breast subset)",
               "n_bcbm_exams": int(len(bcbm_p)), "n_external_breast_exams": int(len(ext_breast)),
               "n_external_control_exams": int((ext_p["is_breast"] == 0).sum()),
               "n_common_features": len(fcols),
               "fraction_features_shifted_fdr05_raw": round(frac_shift, 4),
               "fraction_features_shifted_fdr05_after_location_scale": round(frac_shift_after, 4),
               "median_abs_smd_raw": round(median_smd, 4),
               "model_transfer_external_score_distribution": score_stats,
               "positive_control_breast_vs_other_auroc": pos_ctrl,
               "extraction": {k: EXTRACT[k] for k in ("binCount", "resampledPixelSpacing")},
               "note": "Feature-transfer / domain-shift only; UCSF-BMSR has no ER/PR/HER2 labels, "
                       "so no external receptor AUROC is computable. Per-metastasis (connected "
                       "component) extraction mirrors BCBM per-lesion features. UCSF-BMSR used under "
                       "its DUA; cite Rudie et al., Radiol Artif Intell 2024."}
    (METADATA_DIR / "step18_external_feature_transfer_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Completed %s. shifted raw=%.0f%% -> after align=%.0f%% | median|SMD|=%.2f | "
                "pos_ctrl(breast-vs-other)=%s", STEP_NAME, frac_shift * 100, frac_shift_after * 100,
                median_smd, pos_ctrl)
    logger.info("=" * 80)


def _load_bcbm_labels(case_ids: List[str]) -> Dict[str, np.ndarray]:
    try:
        df = pd.read_csv(LESION_CSV).drop_duplicates("patient_base").set_index("patient_base")
    except Exception:
        return {}
    out = {}
    for tcol in TARGETS:
        if tcol in df.columns:
            out[tcol] = np.array([float(df[tcol].get(cid, np.nan)) if cid in df.index else np.nan
                                  for cid in case_ids], dtype=float)
    return out


if __name__ == "__main__":
    main()
