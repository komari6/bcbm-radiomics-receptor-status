# MRI radiomics does not predict receptor status in breast-cancer brain metastases

Analysis code and derived results for a leakage-controlled radiomics study of oestrogen receptor
(ER), progesterone receptor (PR) and HER2 status in breast-cancer brain metastases (BCBM).

**Headline finding (negative).** Under nested cross-validation in which every preprocessing,
harmonization and feature-selection step is fitted inside the training folds, MRI radiomics does not
predict receptor status: nested-CV AUC ≈ 0.50 for all three targets, indistinguishable from a
permutation null. The locked test set (16–17 patients) is uninformative by construction
(power ≈ 22–40%). This repository releases the full pipeline so that both the result and the
methodological argument can be checked and re-run.

## What is in this repository

| Path | Contents |
| --- | --- |
| `src/` | The analysis pipeline, steps 00–18 (see below) |
| `run_pipeline.py` | Sequential driver for the pipeline steps |
| `results/tables/` | Result tables (CSV) behind every number in the paper |
| `real_files_mapping.xlsx` | Lesion image/mask filename index for the TCIA collection (step 00 input) |
| `results/figures/` | The publication figure package (PNG + PDF) |
| `results/metadata/` | Per-step JSON summaries |
| `docs/` | Pipeline output map, file-structure reference, external-transfer protocol |

Imaging data are **not** included — see *Data access* below.

## Pipeline

| Step | Module | Purpose |
| --- | --- | --- |
| 00–02 | `src/data/` | Build master file, initialize project, audit data |
| 03–04 | `src/data/` | Segmentation harmonization, feature engineering |
| 05 | `src/modeling/` | Tabular radiomics modeling (nested CV) |
| 06–08 | `src/modeling/` | Image baseline, MIL radiomics, CNN |
| 09 | `src/modeling/` | Hybrid fusion |
| 10–11 | `src/external/` | External proxy cohort build and validation |
| 12 | `src/analysis/` | ComBat (empirical Bayes) sensitivity, statistical comparison |
| 13 | `src/reporting/` | Figures and reports |
| 14 | `src/analysis/` | Learning curve and statistical power |
| 15 | `src/analysis/` | Permutation null, seed stability, calibration |
| 16 | `src/analysis/` | Preprocessing sensitivity (requires PyRadiomics) |
| 17 | `src/analysis/` | radMLBench optimism / generalization-gap benchmark |
| 18 | `src/analysis/` | External feature transfer vs UCSF-BMSR (requires PyRadiomics) |

## Data access

### BCBM-RadioGenomics (primary cohort — public)

The imaging dataset is publicly available from The Cancer Imaging Archive under **CC BY 4.0**:
<https://www.cancerimagingarchive.net/collection/bcbm-radiogenomics/>

Required data citation:

> Taha B, Wu D, Sabal L, Kollitz M, Venteicher A, Watanabe Y. *MRI Dataset of Metastatic Breast
> Cancer to the Brain with Expert-reviewed Segmentations and Tumor-derived Radiomic Features*
> (Version 1) [Dataset]. The Cancer Imaging Archive; 2025. <https://doi.org/10.7937/RRSE-W278>

Download the collection yourself; this repository re-hosts no imaging. Step 00 reads the following
from the repository root (bare relative paths — run it from there):

| Input | Where it comes from |
| --- | --- |
| `BCBM-RadioGenomic_Radiomics_Data.xlsx` (sheet `merged_orig`) | TCIA collection |
| `BCBM-RadioGenomics-Clinical-data.xlsx` (sheet `Clinical+Genetics`) | TCIA collection |
| `BCBM-RadioGenomics_Images_Masks_Dec2024/` | TCIA collection (images + masks) |
| `real_files_mapping.xlsx` (sheet `Sheet1`) | **Included in this repository** — maps each patient folder to its image/mask filenames |

Outputs are written to `bcbm_project/reports/` and `bcbm_project/metadata/`; the copies under
`results/` here are those same files, restaged for browsing.

### OpenBTAI (steps 10–11 — public, not re-hosted)

The external *proxy* cohort is OpenBTAI (Open database of Brain Tumors for studies in Artificial
Intelligence), CC BY 4.0. Steps 10–11 read `OpenBTAI_METS_ClinicalData_Nov2023.xlsx`,
`OpenBTAI_RADIOMICS.xlsx`, and `OpenBTAI_MORPHOLOGICAL_MEASUREMENTS.xlsx` from the repository root.
Its receptor labels are *derived from breast molecular subtype*, not pathology assays — the paper
reports this as proxy, not direct external validation.

> Ocaña-Tienda B, Pérez-Beteta J, Villanueva-García JD, et al. A comprehensive dataset of annotated
> brain metastasis MR images with clinical and radiomic data. *Sci Data*. 2023;10:208.
> <https://doi.org/10.1038/s41597-023-02123-0> — dataset: <https://doi.org/10.6084/m9.figshare.20579541>

### UCSF-BMSR (step 18 only — access-controlled)

Step 18 compares feature distributions against the breast subset of the **UCSF Brain Metastases
Stereotactic Radiosurgery** dataset (Rudie et al.). That dataset is released under a **Data Use
Agreement** and is **deliberately absent from this repository**, as are all features derived from it:
the agreement permits use only within the licensee's own organization and forbids onward transfer
without written consent from UCSF. To run step 18 you must obtain UCSF-BMSR under your own DUA. Only
aggregate statistics derived from it — the numbers reported in the paper — appear in `results/`.

## Reproducing

```bash
python -m venv .venv
.venv\Scriptsctivate            # Windows  |  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py             # runs steps 00-18 from the repository root
```

Requires Python ≥ 3.10. Optional dependencies degrade gracefully, but note that two of them change
results rather than merely skipping work: without `xgboost` step 05 falls back to extra trees, and
without `lightgbm` the model selected as best for PR is unavailable. Without `torch` steps 07–09 are
skipped; without `nibabel`/`SimpleITK` the NIfTI-reading steps are skipped; without `seaborn` step 13
uses matplotlib defaults.

### What does and does not regenerate

Given the inputs above, the released code regenerates every result table and every figure **except**:

- **Steps 10–11** — need the OpenBTAI workbooks (see above).
- **Step 18** — needs UCSF-BMSR under its own DUA (see above).

Figures 1 and 2 of the paper are generated by `src/reporting/step_13b_methods_figure.py` and
`src/reporting/step_13c_representative_mri.py`. Neither is part of `run_pipeline.py` — run them
directly. `step_13c` needs the source NIfTIs; it selects, within each display group, the two
lesions nearest that group's median volume among the 340 lesions of at least 0.5 cc, taking only
patients not already shown, so the panel is deterministic.

Step 17 does regenerate (`radMLBench` is in `requirements.txt`); it is pinned to the first 18 of the
collection's 50 datasets (`--max-datasets 18`), the compute-budget cut used for the published numbers.

Seeds are fixed (seed 42). The scikit-learn steps are deterministic; the PyTorch steps (07–09) set
the seed and `cudnn.deterministic` but do not call `torch.use_deterministic_algorithms`, so
deep-model numbers are not guaranteed bitwise identical across hardware.

Steps 16 and 18 need **PyRadiomics**, which does not build against NumPy 2.x or Python ≥ 3.12. Use
the separate environment and invoke it through `conda run` (running its `python.exe` directly leaves
`Library\bin` off `PATH` and NumPy fails to load its DLLs):

```bash
conda env create -f environment_radiomics.yml
conda run -n bcbm_radiomics python src/analysis/step_16_preprocessing_sensitivity.py
```

No GPU is required — the deep-learning steps run on CPU, more slowly.

## Citing this work

If you use this code, please cite the paper (see `CITATION.cff`) and the BCBM-RadioGenomics data
citation above. Work involving step 18 must additionally cite Rudie et al. for UCSF-BMSR.

## License

Code is released under the MIT License (`LICENSE`). The derived result tables and figures in
`results/` are likewise MIT-licensed, but are derived from BCBM-RadioGenomics and therefore carry
that dataset's CC BY 4.0 attribution requirement — please retain the data citation above.
