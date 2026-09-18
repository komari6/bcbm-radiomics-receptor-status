# FILE STRUCTURE

## Source Scripts (project root: `D:\PycharmProjects\bcbm_project\`)

### Orchestrator

| File | Role |
|------|------|
| `run_pipeline.py` | Runs steps 00–13; supports `--from-step`, `--only-step`, `--skip`, `--force-step`, `--no-cache`, `--external`, `--reports-only`, `--dry-run` |

### Data Preparation — `src/data/`

| File | Step | Role | GPU? |
|------|------|------|------|
| `step_00_build_master_file.py`           | 00 | Merges 3 Excel sources → `BCBM_Full_Data.xlsx` | No |
| `step_01_project_initialization.py`     | 01 | Creates `bcbm_project/` folder tree, copies raw data | No |
| `step_02_data_audit.py`                 | 02 | ID validation, NIfTI path reconstruction, canonical CSV | No |
| `step_03_segmentation_harmonization.py` | 03 | Classifies mask descriptors, filters to lesion-only | No |
| `step_04_feature_engineering.py`        | 04 | Patient aggregation, scanner audit, train/valid/test splits | No |

### Internal Modeling — `src/modeling/`

| File | Step | Role | GPU? |
|------|------|------|------|
| `step_05_tabular_modeling.py`       | 05 | Nested CV + fold-safe ComBat + stability selection + ElasticNet/LGBM | No |
| `step_06_image_baseline_modeling.py`| 06 | Shallow image baseline (histogram + PCA + sklearn) | No |
| `step_07_mil_radiomics.py`          | 07 | Attention-MIL on lesion radiomics bags (PyTorch) | Optional |
| `step_08_cnn_image.py`              | 08 | 2.5D CNN-MIL on raw MRI slices (EfficientNet-B0) | Required |
| `step_09_hybrid_fusion.py`          | 09 | Hybrid: radiomics MIL + CNN image stream fusion | Required |

### External Validation — `src/external/`

| File | Step | Role | GPU? |
|------|------|------|------|
| `step_10_build_openbtai_external_proxy_dataset.py` | 10 | Builds OpenBTAI external proxy dataset from 3 Excel files | No |
| `step_11_openbtai_external_proxy_validation.py`    | 11 | Validates internal pipeline on OpenBTAI proxy labels | No |

### Statistical Analysis — `src/analysis/`

| File | Step | Role | GPU? |
|------|------|------|------|
| `step_12_combat_eb_and_statistical_comparison.py` | 12 | EB-ComBat sensitivity + pairwise DeLong/bootstrap AUROC comparison | No |

### Reporting — `src/reporting/`

| File | Step | Role | GPU? |
|------|------|------|------|
| `step_13_figures_and_reports.py` | 13 | All publication figures, tables, manifest, and ZIP package | No |

---

## Generated Project Tree (`./bcbm_project/`)

```
bcbm_project/
├── data/
│   ├── raw/
│   │   ├── BCBM_Full_Data.xlsx              Step 01 copies here from root
│   │   └── BCBM-RadioGenomics_Images_Masks_Dec2024/
│   ├── interim/
│   │   └── tabular_raw_canonical.csv        Step 02 output
│   ├── processed/
│   │   ├── analysis_ready_step03_lesion_only.csv      Step 03 → read by 06,07,08,09
│   │   ├── analysis_ready_step04_patient_level.csv    Step 04 → read by 05,11
│   │   ├── analysis_ready_step04_case_level.csv       Step 04
│   │   └── analysis_ready_step04_patient_level.parquet
│   ├── external_ready/                      Step 10 outputs → read by Step 11
│   │   ├── openbtai_breast_only_patient_level_proxy_ready.csv
│   │   ├── openbtai_breast_only_lesion_level_proxy_ready.csv
│   │   ├── openbtai_clinical_with_receptor_proxy_labels.csv
│   │   ├── openbtai_patient_level_external_proxy_ready.csv
│   │   └── openbtai_feature_dictionary.json
│   └── external_raw/                        (reserved for raw external source files)
│
├── metadata/
│   ├── step04_feature_dictionary.json
│   ├── step04_patient_splits.csv
│   ├── step04_scanner_confounding_audit.json
│   ├── step04_summary.json
│   ├── step05master_summary.json
│   ├── step07_mil_summary.json
│   ├── step08_cnn_summary.json
│   ├── step09_hybrid_summary.json
│   ├── step10_openbtai_build_summary.json
│   ├── step11_openbtai_external_proxy_summary.json
│   ├── step12_summary.json
│   ├── step13_publication_summary.json
│   ├── pipeline_state.json                  StateManager checkpoint
│   └── pipeline_run_manifest.json           Written by run_pipeline.py after each run
│
├── cache/
│   ├── masks/                               Step 03 mask geometry cache
│   ├── image_features/                      Step 06 histogram/PCA cache
│   ├── cnn_embeddings/                      Step 08 CNN embedding cache
│   ├── radiomics_bags/                      Steps 07, 09 bag cache
│   ├── tabular_preprocessing/               Step 05 fold preprocessor cache
│   └── external_mapping/                    Step 10 merge manifest cache
│
├── models/
│   ├── step05_tabular/                      Step 05 fitted sklearn models
│   ├── step07_mil/                          Step 07 PyTorch MIL weights
│   ├── step08_cnn/                          Step 08 CNN + backbone weights
│   ├── step09_hybrid/                       Step 09 fusion model weights
│   └── step11_external_refit/               Step 11 refitted internal model
│
├── reports/
│   ├── tables/
│   │   ├── step05master_*.csv               Step 05 all result tables
│   │   ├── step06_image_benchmark/          Step 06 tables (subdirectory)
│   │   ├── step07_mil_*.csv                 Step 07 result tables
│   │   ├── step08_cnn_*.csv                 Step 08 result tables
│   │   ├── step09_hybrid_*.csv              Step 09 result tables
│   │   ├── step09_final_comparison_all_stages.csv
│   │   ├── step10_openbtai_proxy_label_distribution.csv
│   │   ├── step11_openbtai_external_proxy_*.csv
│   │   ├── step12_ebcombat_sensitivity_results.csv
│   │   ├── step12_locked_test_predictions.csv
│   │   ├── step12_internal_pairwise_model_comparison.csv
│   │   └── step12_external_pairwise_model_comparison.csv
│   └── figures/
│       ├── step05master_*.png
│       ├── step06_image_benchmark/
│       ├── step07_mil_*.png
│       ├── step08_*.png
│       ├── step09_*.png
│       ├── step11_openbtai_*.png
│       ├── step12_*.png
│       └── step13_publication_package/      Final 20 publication figures (PNG + PDF)
│           ├── fig_00_*.{png,pdf}
│           ├── ...
│           ├── fig_15_*.{png,pdf}
│           ├── fig_S1_*.{png,pdf} ... fig_S4_*.{png,pdf}
│           └── figure_manifest.csv
│
├── logs/
│   ├── step_05_tabular_modeling.log
│   ├── step_07_mil_radiomics.log
│   ├── step_08_cnn_image.log
│   ├── step_09_hybrid_fusion.log
│   ├── step_10_build_openbtai_external_proxy_dataset.log
│   ├── step_11_openbtai_external_proxy_validation.log
│   ├── step_12_combat_eb_and_statistical_comparison.log
│   └── step_13_figures_and_reports.log
│
└── checkpoints/
    └── pipeline_state.json
```

---

## Important Notes on Paths

- All step scripts use `PROJECT_ROOT = Path("./bcbm_project").resolve()` — they must be
  invoked with `cwd` set to `D:\PycharmProjects\bcbm_project\` (the directory containing
  `run_pipeline.py`). `run_pipeline.py` sets this automatically via `cwd=str(ROOT)`.
- Step 06 outputs go into **subdirectories** (`step06_image_benchmark/`).
  Step 13 can optionally include Step 06 results in comparison tables.
- Step 10 must run before Step 11. Step 11 reads
  `bcbm_project/data/external_ready/openbtai_breast_only_patient_level_proxy_ready.csv`.
- Step 11 dynamically imports Step 05 utilities via `importlib`. The path is resolved
  relative to `src/external/` → `../modeling/step_05_tabular_modeling.py`.

---

## Documentation Files

| File | Purpose |
|------|---------|
| `CLAUDE_START_HERE.md`              | Navigation index for Claude sessions |
| `PROJECT_CONTEXT.md`                | Clinical background, dataset, methodology |
| `FILE_STRUCTURE.md`                 | This file |
| `PIPELINE_OUTPUTS.md`               | Every input/output per step |
| `CHANGELOG.md`                      | Development log (kept in the project; not part of the public release) |
| `KNOWN_ISSUES.md`                   | Open bugs, cautions, design gaps |
| `CLAUDE_INSTRUCTIONS.md`            | Rules for future Claude edits |
| `BCBM_pipeline_reorganization_plan.md` | Reorganization rationale and design decisions |
