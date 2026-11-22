# Leakage-controlled evaluation of MRI radiomics for receptor status in breast-cancer brain metastases

Analysis code and derived results for a leakage-controlled radiomics study of oestrogen receptor
(ER), progesterone receptor (PR) and HER2 status in breast-cancer brain metastases (BCBM).

**Headline finding.** Under repeated nested cross-validation in which every preprocessing,
harmonization and feature-selection step is fitted inside the training folds, MRI radiomics did not
demonstrate reproducible discrimination of the available patient-level receptor label: mean AUROC
0.47 (ER), 0.46 (PR) and 0.57 (HER2), and no receptor exceeded a permutation null of the full
procedure. This is not evidence that receptor status cannot be predicted from brain-metastasis
imaging; the patient-bootstrap intervals do not exclude a modest effect. Splitting lesions of one
patient across training and test data raised AUROC to about 0.78 with nothing else changed. The
locked test set (16–17 patients) is too small to be informative. This repository releases the full
pipeline so that both the result and the methodological argument can be checked and re-run.

## What is in this repository

| Path | Contents |
| --- | --- |
| `src/` | The analysis pipeline, steps 00–24 (see below) |
| `run_pipeline.py` | Sequential driver for steps 00–18 |
| `results/tables/` | Result tables (CSV) behind every number in the paper |
| `results/metadata/` | Per-step JSON summaries and the patient-level train/tuning/test split |
| `results/partitions/` | Fold assignment of every patient in every repetition of the cross-validation |
| `results/figures/` | The figure package written by step 13 (PNG + PDF) |
| `real_files_mapping.xlsx` | Lesion image/mask filename index for the TCIA collection (step 00 input) |
| `docs/` | Analyses on record, analysis protocol, pipeline output map, file-structure reference |
| `tools/` | Print-size figure renderers and a helper that rebuilds the step 09 cross-stage table |

Imaging data are **not** included — see *Data access* below.

## Pipeline

| Step | Module | Purpose |
| --- | --- | --- |
| 00–02 | `src/data/` | Build master file, initialize project, audit data |
| 03–04 | `src/data/` | Segmentation harmonization (lesion set), feature engineering |
| 05 | `src/modeling/` | Tabular radiomics modeling, single locked split |
| 05b | `src/modeling/` | Repeated nested cross-validation, the primary analysis |
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
| 19 | `src/analysis/` | Unit of analysis: random versus patient-grouped splitting of lesions and imaging cases |
| 20 | `src/analysis/` | Patients analyzed versus not analyzed |
| 21 | `src/analysis/` | Detectable effect with patients as the unit |
| 22 | `src/analysis/` | Attenuation of AUROC by label error (simulation) |
| 23 | `src/analysis/` | Fold assignments, re-derived from the seeds |
| 24 | `src/analysis/` | Full-procedure permutation null, reduced pipeline on ten partitions, lesion arms capped at 20 lesions per patient, one-sided label error |

Steps 19–23 follow the protocol in `docs/PROTOCOL_added_analyses.md`, locked before they were run;
step 24 was added afterwards and is reported whatever its direction. `docs/ANALYSES.md` lists the
command, outputs and result of every analysis.

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
reports this as an attempted proxy analysis, not external validation.

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
.venv\Scripts\activate            # Windows  |  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py             # runs steps 00-18 from the repository root
python src/modeling/step_05b_repeated_nested_cv.py --jobs 10     # primary analysis (Table 2)
```

Steps 19–24 are run directly, after steps 00–05b, with the commands given in each module's docstring
and in `docs/ANALYSES.md`. Step 09 reads the result files of step 05, so run it after step 05 has
finished; if it was run earlier, `python tools/rebuild_step09_comparison.py` rewrites its cross-stage
table from the stored results without refitting anything.

Requires Python ≥ 3.10. Optional dependencies degrade gracefully, but note that two of them change
results rather than merely skipping work: without `xgboost` step 05 falls back to extra trees, and
without `lightgbm` part of the model pool is unavailable. Without `torch` steps 07–09 are skipped;
without `nibabel`/`SimpleITK` the NIfTI-reading steps are skipped; without `seaborn` step 13 uses
matplotlib defaults.

### What does and does not regenerate

Given the inputs above, the released code regenerates every result table and every figure **except**:

- **Steps 10–11** — need the OpenBTAI workbooks (see above).
- **Step 18** — needs UCSF-BMSR under its own DUA (see above).

The workflow schematic and the representative-MRI panel are generated by
`src/reporting/step_13b_methods_figure.py` and `src/reporting/step_13c_representative_mri.py`.
Neither is part of `run_pipeline.py` — run them directly. `step_13c` needs the source NIfTIs; it selects, within each display group, the two
lesions nearest that group's median volume among the 295 analyzed lesions of at least 0.5 cc, taking
only patients not already shown, so the panel is deterministic.

### Print-size figures

The figures as printed in the article and its supporting information are drawn at their final size
by `tools/render_print_figures.py` (the eight article figures) and `tools/render_print_supplement.py`
(the supporting figures, one per page): Arial, no text below 8 pt, no title inside the graphic.
Both read only the result files the pipeline has written, so run the pipeline (and `step_13c` for
the MRI panel) first; they re-run no analysis, and `render_print_figures.py` asserts the values it
plots against the result tables. Output goes to `results/figures/print_article/` and
`results/figures/print_supplement/`, or to `--out DIR`. The external feature-transfer figure is
taken from step 18's stored image, because its principal-component coordinates are not stored.

Step 17 does regenerate (`radMLBench` is in `requirements.txt`); it is pinned to the first 18 of the
collection's 50 datasets (`--max-datasets 18`), the compute-budget cut used for the published numbers.

Seeds are fixed (seed 42). The scikit-learn steps are deterministic; the PyTorch steps (07–09) set
the seed and `cudnn.deterministic` but do not call `torch.use_deterministic_algorithms`, so
deep-model numbers are not guaranteed bitwise identical across hardware. The published numbers come
from a CPU run.

Steps 16 and 18 need **PyRadiomics**, which does not build against NumPy 2.x or Python ≥ 3.12. Use
the separate environment and invoke it through `conda run` (running its `python.exe` directly leaves
`Library\bin` off `PATH` and NumPy fails to load its DLLs):

```bash
conda env create -f environment_radiomics.yml
conda run -n bcbm_radiomics python src/analysis/step_16_preprocessing_sensitivity.py
```

No GPU is required — the deep-learning steps run on CPU, more slowly.

## Changes in version 1.2.0

- **Lesion set (step 03).** Masks that derive from another mask of the same lesion (margin and 1-mm
  expansions, second contours, overlapping re-exports, within-examination duplicates by voxel
  overlap) are now classified as `lesion_derivative` instead of lesions, and eight tumor masks with
  structure names are classified as lesions. The analysis now covers 1,522 lesions in 140 patients
  (1,841 in 139 in version 1.1.0); every step was rerun on the corrected set.
- **Locked test kept out of the robustness analyses.** Steps 15 and 16 now use the training and
  tuning patients only, and step 18 blanks locked-test labels before its transfer model.
- **Step 04** refreshes the stored patient split when a split is reused.
- **New steps 19–24**, the analysis protocol and the fold assignments (see above).
- The figure composites of version 1.1.0 (`fig_SC*`), built from the earlier results, are not
  included; `tools/build_supp_composites.py` rebuilds them from the current figures.

## Citing this work

If you use this code, please cite the paper (see `CITATION.cff`) and the BCBM-RadioGenomics data
citation above. Work involving step 18 must additionally cite Rudie et al. for UCSF-BMSR.

## License

Code is released under the MIT License (`LICENSE`). The derived result tables and figures in
`results/` are likewise MIT-licensed, but are derived from BCBM-RadioGenomics and therefore carry
that dataset's CC BY 4.0 attribution requirement — please retain the data citation above.
