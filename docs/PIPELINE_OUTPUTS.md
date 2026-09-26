# PIPELINE OUTPUTS

> **Rule: Do not rename any output file listed here without updating this document.**
> Steps 09, 11, 12, and 13 read these paths by exact name. A silent rename = silently skipped output.

---

## Step 00 — `src/data/step_00_build_master_file.py`

**Inputs (in script working directory = project root):**
- `BCBM-RadioGenomic_Radiomics_Data.xlsx` (sheet: `merged_orig`)
- `BCBM-RadioGenomics-Clinical-data.xlsx` (sheet: `Clinical+Genetics`)
- `real_files_mapping.xlsx` (sheet: `Sheet1`)

**Outputs (in project root — Step 01 copies to `bcbm_project/data/raw/`):**
- `BCBM_Full_Data.xlsx` — merged master dataset
- `BCBM_Master_Merged_Missing_Report.csv`
- `BCBM_Master_Merged_Duplicate_Key_Report.csv`
- `BCBM_Master_Merged_Audit.json`

---

## Step 01 — `src/data/step_01_project_initialization.py`

**Inputs:**
- `BCBM_Full_Data.xlsx`
- `BCBM-RadioGenomics_Images_Masks_Dec2024/` (raw NIfTI folder)

**Outputs:**
- Full `bcbm_project/` folder tree (all subdirectories including `cache/`, `models/`, `external_ready/`)
- `bcbm_project/data/raw/BCBM_Full_Data.xlsx`
- `bcbm_project/metadata/project_config.json`
- `bcbm_project/metadata/pipeline_state.json`

---

## Step 02 — `src/data/step_02_data_audit.py`

**Inputs:**
- `bcbm_project/data/raw/BCBM_Full_Data.xlsx`

**Outputs:**
- `bcbm_project/data/interim/tabular_raw_canonical.csv` — validated, path-resolved table
- `bcbm_project/metadata/step02_data_audit_summary.json`
- `bcbm_project/reports/tables/step02_path_resolution_report.csv`

**Cache:**
- `bcbm_project/cache/external_mapping/internal_path_index.parquet`

---

## Step 03 — `src/data/step_03_segmentation_harmonization.py`

**Inputs:**
- `bcbm_project/data/interim/tabular_raw_canonical.csv`

**Outputs (critical — read by steps 06, 07, 08, 09):**
- `bcbm_project/data/processed/analysis_ready_step03_lesion_only.csv`
  - Key columns: `patient_base`, `case_id`, `image_abs_path`, `mask_abs_path`, lesion-level radiomics features
- `bcbm_project/data/processed/analysis_ready_step03_full.csv`
- `bcbm_project/metadata/step03_summary.json`
- `bcbm_project/reports/tables/step03_mask_classification_report.csv`

**Cache:**
- `bcbm_project/cache/masks/mask_geometry_features.parquet`
- `bcbm_project/cache/masks/mask_geometry_manifest.json`

---

## Step 04 — `src/data/step_04_feature_engineering.py`

**Inputs:**
- `bcbm_project/data/processed/analysis_ready_step03_lesion_only.csv`

**Outputs (critical — read by steps 05–13):**
- `bcbm_project/data/processed/analysis_ready_step04_patient_level.csv`
  - Key columns: `patient_base`, `split` (train/valid/test), `target_er`, `target_pr`, `target_her2`, all aggregated radiomics features, `dominant_manufacturer`, `dominant_field_strength_t`
- `bcbm_project/data/processed/analysis_ready_step04_case_level.csv`
- `bcbm_project/metadata/step04_feature_dictionary.json` — lists radiomics feature names by family
- `bcbm_project/metadata/step04_patient_splits.csv` — patient → split assignment
- `bcbm_project/metadata/step04_summary.json`
- `bcbm_project/metadata/step04_scanner_confounding_audit.json`

---

## Step 05 — `src/modeling/step_05_tabular_modeling.py`

**Inputs:**
- `bcbm_project/data/processed/analysis_ready_step04_patient_level.csv`
- `bcbm_project/metadata/step04_feature_dictionary.json`

**Outputs — master tables (read by steps 09, 11, 12, 13):**

| File | Description |
|------|-------------|
| `reports/tables/step05master_model_results.csv` | All test results (one row per target × variant × model) |
| `reports/tables/step05master_global_best_models_by_target.csv` | Best model per target (by nested CV AUROC) |
| `reports/tables/step05master_feature_set_variant_summary.csv` | Best AUROC per target × variant |
| `reports/tables/step05master_scenario_summary.csv` | Summary by scenario |
| `reports/tables/step05master_inner_cv_comparison_master.csv` | Inner-CV model selection table |
| `reports/tables/step05master_validation_model_comparison_master_table.csv` | Alias of above (Step 13 compatibility) |

**Outputs — per target × variant (pattern):**
- `reports/tables/step05master_{target}_{variant}_inner_cv_comparison.csv`
- `reports/tables/step05master_{target}_{variant}_best_model_test_result.csv`
- `reports/tables/step05master_{target}_{variant}_{model}_feature_importance_native.csv`
- `reports/tables/step05master_{target}_{variant}_{model}_feature_importance_permutation.csv`
- `reports/tables/step05master_{target}_{variant}_{model}_selected_features.csv`

**Outputs — metadata:**
- `metadata/step05master_summary.json`

**Models saved:**
- `models/step05_tabular/{target}/{variant}/{model_name}.joblib`
- `models/step05_tabular/model_registry_step05.json`

**Cache:**
- `cache/tabular_preprocessing/step05_fold_preprocessors.joblib`
- `cache/tabular_preprocessing/step05_selected_features_by_fold.json`

**Key columns in `step05master_model_results.csv`:**
`target`, `feature_set_variant`, `model`, `model_display`, `scenario`, `nested_cv_auroc_mean`, `nested_cv_auroc_std`, `AUROC`, `AUPRC`, `F1`, `Balanced_Acc`, `Recall_Sensitivity`, `Specificity`, `Precision`, `TN`, `FP`, `FN`, `TP`, `validation_AUROC_at_selection`, `generalization_gap_AUROC`

> ⚠️ Step 05 uses `Recall_Sensitivity` (not `Sensitivity`). Step 13's `ensure_result_columns` maps this correctly.

**Feature set variants:** `radiomics_plus_scanner`, `radiomics_only`, `scanner_only`

---

## Step 06 — `src/modeling/step_06_image_baseline_modeling.py`

**Inputs:**
- `bcbm_project/data/processed/analysis_ready_step03_lesion_only.csv`
- `bcbm_project/data/processed/analysis_ready_step04_patient_level.csv`
- `bcbm_project/metadata/step04_feature_dictionary.json`

**Outputs (shallow benchmark — optional inclusion in Step 13):**
- `reports/tables/step06_image_benchmark/step06_model_results.csv`
- `reports/tables/step06_image_benchmark/step06_validation_model_comparison_master_table.csv`
- `reports/tables/step06_image_benchmark/step06_lesion_image_feature_table.csv`
- `reports/tables/step06_image_benchmark/step06_patient_level_image_augmented_table.csv`
- `reports/figures/step06_image_benchmark/` — various plots
- `metadata/step06_image_baseline_summary.json`

**Cache:**
- `cache/image_features/step06_histogram_features.parquet`
- `cache/image_features/step06_pca_features.joblib`

**Modality variants:** `image_only`, `radiomics_only`, `scanner_only`, `image_plus_radiomics`, `hybrid_all`

---

## Step 07 — `src/modeling/step_07_mil_radiomics.py`

**Inputs:**
- `bcbm_project/data/processed/analysis_ready_step03_lesion_only.csv`
- `bcbm_project/metadata/step04_patient_splits.csv`
- `bcbm_project/metadata/step04_feature_dictionary.json`

**Outputs (read by steps 09, 13):**

| File | Description |
|------|-------------|
| `reports/tables/step07_mil_model_results.csv` | All test results (prefixed `test_`) |
| `reports/tables/step07_mil_best_by_target.csv` | Best model per target (by valid composite score) |
| `reports/tables/step07_mil_best_by_target_test_audit_only.csv` | Audit-only test-best table |
| `reports/tables/step07_mil_predictions.csv` | Patient-level predictions |
| `reports/tables/step07_mil_attention_weights.csv` | Lesion attention weights |
| `metadata/step07_mil_summary.json` | Config + result summary |

**Models saved:**
- `models/step07_mil/{target}/best_mil_model.pt`
- `models/step07_mil/{target}/scaler_imputer.joblib`
- `models/step07_mil/model_registry_step07.json`

**Cache:**
- `cache/radiomics_bags/step07_radiomics_bags.npz`
- `cache/radiomics_bags/step07_bag_index.csv`

**Config class:** `Step08AConfig` (historical name)
**Models:** `attention_mil`, `gated_attention_mil`

---

## Step 08 — `src/modeling/step_08_cnn_image.py`

**Inputs:**
- `bcbm_project/data/processed/analysis_ready_step03_lesion_only.csv`
- `bcbm_project/metadata/step04_patient_splits.csv`
- NIfTI files (via `image_abs_path`, `mask_abs_path` columns)

**Outputs (read by steps 09, 13):**

| File | Description |
|------|-------------|
| `reports/tables/step08_cnn_model_results.csv` | Test results per target (prefixed `test_`) |
| `reports/tables/step08_cnn_predictions.csv` | Patient-level predictions |
| `metadata/step08_cnn_summary.json` | Config + result summary |

**Models saved:**
- `models/step08_cnn/{target}/best_cnn_mil_model.pt`
- `models/step08_cnn/{target}/cnn_backbone_state.pt`
- `models/step08_cnn/model_registry_step08.json`

**Cache (critical — reused by Step 09):**
- `cache/cnn_embeddings/step08_cnn_embeddings.npz`
- `cache/cnn_embeddings/step08_cnn_embeddings_index.csv`
- Compatibility copy: `metadata/step08_cnn_embeddings.npz` + `metadata/step08_cnn_embeddings_index.csv`

**Config class:** `Step08BConfig`
**Architecture:** EfficientNet-B0 + Gated Attention-MIL
**GPU:** Required (CUDA). AMP enabled when `use_amp=True` and CUDA available.

---

## Step 09 — `src/modeling/step_09_hybrid_fusion.py`

**Inputs:**
- `bcbm_project/data/processed/analysis_ready_step03_lesion_only.csv`
- `bcbm_project/metadata/step04_patient_splits.csv`
- `bcbm_project/metadata/step04_feature_dictionary.json`
- `bcbm_project/cache/cnn_embeddings/step08_cnn_embeddings.npz` (from Step 08)
- NIfTI files (optional — disables image stream if unavailable)
- Steps 05/07/08 result CSVs (for cross-stage comparison table)

**Outputs (read by Step 13):**

| File | Description |
|------|-------------|
| `reports/tables/step09_hybrid_model_results.csv` | Test results per target (prefixed `test_`) |
| `reports/tables/step09_hybrid_predictions.csv` | Patient-level predictions |
| `reports/tables/step09_final_comparison_all_stages.csv` | **Cross-stage table** (Steps 05/07/08/09) |
| `metadata/step09_hybrid_summary.json` | Config + result summary |

**Models saved:**
- `models/step09_hybrid/{target}/{variant}/best_model.pt`
- `models/step09_hybrid/model_registry_step09.json`

**Cache:**
- `cache/radiomics_bags/step09_radiomics_stream_cache.npz`
- `cache/cnn_embeddings/step09_image_stream_cache.npz`

**Config class:** `Step08CConfig`
**Stage values in cross-stage table:** `Step05_Radiomics`, `Step08A_MIL`, `Step08B_CNN`, `Step09_Hybrid`
(Step 13 also accepts legacy aliases: `Step08C_Hybrid`, `Hybrid`)

---

## Step 10 — `src/external/step_10_build_openbtai_external_proxy_dataset.py`

**Inputs (in project root):**
- `OpenBTAI_METS_ClinicalData_Nov2023.xlsx`
- `OpenBTAI_RADIOMICS.xlsx`
- `OpenBTAI_MORPHOLOGICAL_MEASUREMENTS.xlsx`

**Outputs (read by Step 11):**

| File | Description |
|------|-------------|
| `bcbm_project/data/external_ready/openbtai_breast_only_patient_level_proxy_ready.csv` | **Primary file** — read by Step 11 |
| `bcbm_project/data/external_ready/openbtai_breast_only_lesion_level_proxy_ready.csv` | Lesion-level breast subset |
| `bcbm_project/data/external_ready/openbtai_clinical_with_receptor_proxy_labels.csv` | Clinical with proxy labels |
| `bcbm_project/data/external_ready/openbtai_patient_level_external_proxy_ready.csv` | All-tumor patient-level |
| `bcbm_project/data/external_ready/openbtai_feature_dictionary.json` | Feature list for Step 11 |
| `bcbm_project/metadata/step10_openbtai_build_summary.json` | Summary metrics |
| `bcbm_project/reports/tables/step10_openbtai_proxy_label_distribution.csv` | Label counts per proxy target |
| `bcbm_project/cache/external_mapping/openbtai_merge_manifest.json` | Merge audit cache |

**Scientific note:** Labels are subtype-derived proxies (not direct receptor assays).
- Luminal A/B → ER=1, PR=1, HER2=0
- Triple Negative → ER=0, PR=0, HER2=0
- HER2-enriched → HER2=1, ER/PR=NA

---

## Step 11 — `src/external/step_11_openbtai_external_proxy_validation.py`

**Inputs:**
- `bcbm_project/data/processed/analysis_ready_step04_patient_level.csv`
- `bcbm_project/metadata/step04_feature_dictionary.json`
- `bcbm_project/reports/tables/step05master_model_results.csv`
- `bcbm_project/reports/tables/step05master_global_best_models_by_target.csv`
- `bcbm_project/data/external_ready/openbtai_breast_only_patient_level_proxy_ready.csv` ← from Step 10

**Outputs (read by steps 12, 13):**

| File | Description |
|------|-------------|
| `reports/tables/step11_openbtai_external_proxy_results.csv` | AUROC + metrics per target |
| `reports/tables/step11_openbtai_external_proxy_predictions.csv` | Patient-level predictions |
| `reports/tables/step11_openbtai_feature_mapping.csv` | BCBM↔OpenBTAI feature alignment |
| `metadata/step11_openbtai_external_proxy_summary.json` | Config + result summary |
| `reports/figures/step11_openbtai_*.png` | ROC/PR curves |

**Models saved:**
- `models/step11_external_refit/{target}/refit_internal_model_for_external.joblib`

**Methodology:** Refit on BCBM train+valid only; test set and OpenBTAI never touch training.

---

## Step 12 — `src/analysis/step_12_combat_eb_and_statistical_comparison.py`

**Inputs:**
- Step 05 outputs (`step05master_model_results.csv`, `step05master_global_best_models_by_target.csv`)
- Step 11 predictions (`step11_openbtai_external_proxy_predictions.csv`)
- `analysis_ready_step04_patient_level.csv`
- `step04_feature_dictionary.json`

**Outputs (read by Step 13):**

| File | Description |
|------|-------------|
| `reports/tables/step12_ebcombat_sensitivity_results.csv` | No-ComBat vs Simple vs EB-ComBat comparison |
| `reports/tables/step12_locked_test_predictions.csv` | Locked test set predictions |
| `reports/tables/step12_internal_pairwise_model_comparison.csv` | DeLong/bootstrap pairwise AUROC (internal); includes Benjamini-Hochberg FDR columns `bootstrap_p_two_sided_fdr_bh`, `delong_p_fdr_bh` (corrected within each scope×target family) |
| `reports/tables/step12_external_pairwise_model_comparison.csv` | DeLong/bootstrap pairwise AUROC (external); includes the same BH-FDR-corrected p-value columns |
| `reports/figures/step12_*.png` | Sensitivity/comparison plots |
| `metadata/step12_summary.json` | Summary |

---

## Step 13 — `src/reporting/step_13_figures_and_reports.py`

**Inputs:** All outputs from steps 03–12 (graceful fallback if missing).

**Outputs:**
- `reports/figures/step13_publication_package/fig_00_*.{png,pdf}` through `fig_15_*.{png,pdf}`
- `reports/figures/step13_publication_package/fig_S1_*.{png,pdf}` through `fig_S4_*.{png,pdf}`
- `reports/figures/step13_publication_package/fig_SC1_*.png` through `fig_SC7_*.png` — the supplement's seven composites, tiled from the figures above by `tools/build_supp_composites.py`; rebuild them after any figure they contain changes
- `reports/figures/step13_publication_package/figure_manifest.csv`
- `metadata/step13_publication_summary.json`
- `reports/step13_publication_figures.zip`

**Figure inventory (20 figures):**

| Figure | Section | Primary data source |
|--------|---------|---------------------|
| fig_00 | Graphical Abstract | step03, step04, step05 |
| fig_01 | Methods | step03, step04 |
| fig_02 | Methods | step04 |
| fig_03 | Methods | step04, step05 (scanner_only) |
| fig_04 | Radiomics Features | step05 summary (native importance) |
| fig_05 | Radiomics Features | step04, step05 summary |
| fig_06 | Model Performance | step05, step07 |
| fig_07 | Model Performance | step04, step05, step07 |
| fig_08 | Model Performance | step05, step07 |
| fig_09 | Model Performance | step05 |
| fig_10 | Model Performance | step05 validation table |
| fig_11 | MIL / Multi-lesion | step03 |
| fig_12 | MIL / Multi-lesion | step05, step07 |
| fig_13 | Discussion | step05 results |
| fig_14 | Discussion | step05 summary (native + perm importance) |
| fig_15 | Model Performance | step09 cross-stage comparison |
| fig_S1 | Supplementary | step03 |
| fig_S2 | Supplementary | step03, step04 |
| fig_S3 | Supplementary | step04 |
| fig_S4 | Supplementary | step04 |

Note: `fig_methods_pipeline.{png,pdf}` (methodology workflow diagram, used as manuscript Figure 1) also lives in the step13 package dir but is generated separately, not by step 13.

---

## Step 14 — `src/analysis/step_14_learning_curve_power.py`

**Inputs:** `data/processed/analysis_ready_step04_patient_level.csv`, `metadata/step04_feature_dictionary.json`; reuses Step 05's leakage-safe pipeline via importlib.

**Outputs:**
| File | Description |
|------|-------------|
| `reports/figures/step13_publication_package/fig_learning_curve.{png,pdf}` | OOF AUROC vs training-set size, per target (manuscript Fig 8) |
| `reports/figures/step13_publication_package/fig_power_analysis.{png,pdf}` | Hanley–McNeil power + required-n (manuscript Fig 9) |
| `reports/tables/step14_learning_curve.csv` | Learning-curve points (fraction, n, AUROC mean/SD) |
| `reports/tables/step14_power_analysis.csv` | Power, MDA, required-n per target × evaluation set |
| `metadata/step14_learning_curve_power_summary.json` | Full summary |

**Purpose:** Distinguish absent signal from insufficient sample size. Learning curves are flat near chance; the nested-CV OOF pool (~116) is adequately powered (0.95–0.98) to detect AUROC 0.70, whereas the locked test (16–17) has ~25% power. Flags: `--force`, `--no-cache` accepted (no-op cache; always recomputes).

---

## Step 05b — `src/modeling/step_05b_repeated_nested_cv.py`

**Inputs:** `--input` patient-level table (default: the primary one), `--feature-dict` matching dictionary.

**Outputs** (every file carries `--tag`, default `primary`):
| File | Description |
|------|-------------|
| `reports/tables/step05b_repeated_nested_cv_<tag>_summary.csv` | Mean, SD, range of AUROC over the repeated partitions, per target × feature block (manuscript Table 2) |
| `reports/tables/step05b_repeated_nested_cv_<tag>_per_repeat.csv` | One row per partition, with `n_features` and `n_trainval` (used for paired comparisons and for provenance) |
| `reports/tables/step05b_patient_bootstrap_<tag>.csv` | Cross-fitted AUROC and its patient-bootstrap interval |
| `reports/tables/step05b_crossfitted_scores_<tag>.csv` | One averaged out-of-fold score per patient (read by Figures S4, S5) |
| `metadata/step05b_repeated_nested_cv_<tag>.json` | Run record: input path, repeats, splits, seed base |

**Purpose:** the primary endpoint. Repeated (10×) five-outer × three-inner nested cross-validation with
all preprocessing inside the training fold. Tags in use: `primary`, `onesession`, `acqfix`,
`onesession_acqfix`, `agg_median`, `agg_largest`, `rocscores`.

---

## Step 15 — `src/analysis/step_15_robustness_null.py`

**Inputs:** `--input` patient table, `--feature-dict`; reuses Step 05's leakage-safe components.

**Outputs** (suffixed by `--tag`, empty for the primary run):
| File | Description |
|------|-------------|
| `reports/tables/step15_robustness<tag>.csv` | Observed OOF AUROC, permutation null mean and 95th percentile, permutation P, count at or above the observed value, its Monte Carlo SE, seed mean/SD, Brier (manuscript Table S6) |
| `reports/figures/step13_publication_package/fig_permutation_null<tag>.{png,pdf}` | Manuscript Figure S6, bottom |
| `metadata/step15_robustness_summary<tag>.json` | Run record, including the permutation count |

**Purpose:** label-permutation null (`--n-permutations`, default 10,000), seed stability (25 seeds) and
probabilistic accuracy. Permutations are drawn serially per target before being scheduled across
`--jobs` workers, so the null does not depend on scheduling and a smaller count reproduces the leading
draws of a larger one.

---

## Steps 16–18 — sensitivity and external analyses

| File | Description |
|------|-------------|
| `reports/tables/step16_preprocessing_sensitivity.csv` | OOF AUROC under five radiomic extraction settings (Table S4) |
| `reports/tables/step17_optimism_gap.csv` | Favorable-single-split optimism across 18 radMLBench datasets (Figure S7, top; the per-dataset values are released here rather than tabulated in the supplement) |
| `reports/tables/step18_external_feature_transfer.csv` | Feature-distribution shift against UCSF-BMSR (Figure S7, bottom) |

---

## Sensitivity runs and file naming

A sensitivity run must never overwrite a primary output.

- **Step 04** writes `*_onesession` (one examination per patient) or `*_agg_median` / `*_agg_largest`
  (`--aggregation`) for all of its outputs — patient and case tables, feature dictionary, splits,
  summary, scanner audit, screening CSV, missingness and label reports, config and runtime record.
  Such a run also leaves `checkpoints/pipeline_state.json` pointing at the primary files.
- **Steps 05b and 15** carry `--tag` on every output.
- When a tagged run replaces a primary result, the previous file is kept as `<name>.pre_<tag>`.
- `docs/ANALYSES.md` records every run that produced a published number, with its command and result.
