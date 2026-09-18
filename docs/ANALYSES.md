# Analyses on record

Every analysis that produced a number in the manuscript is listed here with the command that
produced it, its inputs, its outputs and its headline result.

**Rule: do not re-run an analysis to recover a number. Read it back from the file named below.**
Re-run only when something genuinely changed — the feature dictionary, the input table, or the code
path that computes the result. When that happens, add a row here rather than editing an old one.

## How runs are kept apart

- A sensitivity run never writes to the primary outputs. `step_05b` and `step_15` take `--tag`;
  `step_04` appends a suffix (`_onesession`, `_agg_median`, `_agg_largest`). Every output of a
  tagged run carries that tag.
- When a tagged run replaces a primary result, the previous primary file is kept beside it as
  `<name>.pre_<tag>` (`*.pre_acqfix`, `*.pre_perm10k`). Nothing is deleted.
- Every published number was re-derived from these files by an automated check before
  submission, including that each result file was computed from the current feature dictionary.

## Primary analyses

| Analysis | Command | Input | Key outputs | Date | Result |
|---|---|---|---|---|---|
| **Feature engineering** (patient table, 642/21/79 blocks) | `python src/data/step_04_feature_engineering.py --force` | step 03 lesion table | `data/processed/analysis_ready_step04_patient_level.csv`, `metadata/step04_feature_dictionary.json` | 2026-09-11 | 139 patients, 742 candidate columns |
| **Repeated nested CV** (primary endpoint, Table 2) | `python src/modeling/step_05b_repeated_nested_cv.py --jobs 10` | primary patient table | `step05b_repeated_nested_cv_primary_{summary,per_repeat}.csv`, `step05b_patient_bootstrap_primary.csv` | 2026-09-08, acquisition and burden rows re-run 2026-09-10 (`acqfix`) and merged in | Radiomic block 0.46 / 0.48 / 0.55 (ER/PR/HER2); every interval spans 0.50 |
| **Single locked split** (Table 2's locked-test column, Figs 3, 4, S6, S9) | `python src/modeling/step_05_tabular_modeling.py --force` | primary patient table | `step05master_*` (153 files) | 2026-09-11 (~3 h) | Maximum 0.78 (HER2, acquisition); unaffected arms byte-identical to the previous run |
| **Cross-fitted patient scores** (Figs S4, S5) | `step_05b … --tag rocscores` | primary patient table | `step05b_crossfitted_scores_rocscores.csv` | 2026-09-09 | One averaged out-of-fold probability per patient |
| **Scanner-confounding audit** (patient level) | inside step 04 | step 03 lesion table + patient table | `metadata/step04_scanner_confounding_audit.json`, `step04_scanner_radiomics_shift_screening.csv` | 2026-09-11 | 8 of 107 features differ by field strength, 6 after BH |
| **Deep models** (Table S1, S7) | `step_07` (MIL), `step_08` (CNN), `step_09` (fusion) | lesion images + radiomics | `step07_mil_model_results.csv`, `step08_cnn_model_results.csv`, `step09_final_comparison_all_stages.csv` | 2026-09-06, step 09 2026-09-09 | Trained once, evaluated once; exploratory |
| **EB-ComBat and pairwise tests** (Tables S2, S10) | `python src/analysis/step_12_combat_eb_and_statistical_comparison.py` | locked-test predictions | `step12_ebcombat_sensitivity_results.csv`, `step12_internal_pairwise_model_comparison.csv` | 2026-09-09 | No pairwise difference significant after BH within each receptor |
| **Learning curve and power** (Fig 6, Table S3) | `python src/analysis/step_14_learning_curve_power.py` | primary patient table | `step14_learning_curve.csv`, `step14_power_analysis.csv` | 2026-09-09 | Nominal power 0.95–0.98 for AUROC 0.70 under an independence approximation |
| **Preprocessing sensitivity** (Table S4) | `python src/analysis/step_16_preprocessing_sensitivity.py` | lesion radiomics | `step16_preprocessing_sensitivity.csv` | 2026-09-06 | Out-of-fold AUROC stays within 0.48–0.54 across five extraction settings |
| **radMLBench optimism** (Fig S7, top; per-dataset values released, not tabulated) | `python src/analysis/step_17_radmlbench_generalization.py` | 18 public datasets | `step17_optimism_gap.csv` | 2026-09-06 | Favorable-single-split optimism, median 0.17 |
| **External proxy / feature transfer** (Figure S7, bottom) | `step_11` (OpenBTAI), `step_18` (UCSF-BMSR) | external cohorts | `step11_openbtai_external_proxy_results.csv`, `step18_external_feature_transfer.csv` | 2026-09-09 / 2026-09-06 | Proxy too small to interpret; 93% of features shifted between cohorts |

## Sensitivity analyses

| Question | Command | Outputs | Date | Runtime | Result |
|---|---|---|---|---|---|
| **One examination per patient** (Table S5) | `step_04 --one-session --reuse-split-from <primary table>`, then `step_05b --input …_onesession.csv --feature-dict …_onesession.json --tag onesession` | `*_onesession*` | 2026-09-09, re-run 2026-09-11 (`onesession_acqfix`, 93 min) | 93 min | Acquisition block falls to chance (0.49/0.51/0.51) and loses its lead |
| **Median aggregation** (Table S9) | `step_04 --aggregation median --reuse-split-from <primary table>`, then `step_05b --input …_agg_median.csv --feature-dict …_agg_median.json --variants radiomics_pure --tag agg_median --jobs 10` | `*_agg_median*` (107-column radiomic block) | 2026-09-11 | 54 min | 0.49 / 0.52 / 0.56; every interval spans 0.50 |
| **Largest lesion only** (Table S9) | as above with `--aggregation largest` / `--tag agg_largest` | `*_agg_largest*` | 2026-09-11 | 76 min | 0.46 / 0.52 / 0.55; every interval spans 0.50 |
| **Label permutation, 10,000** (Table S6, Fig S6) | `python src/analysis/step_15_robustness_null.py --n-permutations 10000 --jobs 14 --tag _perm10k` | `step15_robustness.csv` (promoted from `_perm10k`), `fig_permutation_null.png` | 2026-09-12 | 4 h 52 min pooled, 6 h 37 min one-session | Pooled P .12 / .02 / .07; BH .12 / .06 / .10; one examination .26 / .57 / .52 |
| **Permutation regression test** (proves the parallel code reproduces the old table) | `step_15 … --n-permutations 200 --jobs 12 --tag _regress200` | `step15_robustness_regress200.csv` | 2026-09-11 | 9 min | Identical to the 200-permutation table on every original column |
| **Acquisition block after removing mask-derived columns** | `step_05b … --variants acquisition_only,radiomics_plus_burden --tag acqfix --jobs 10` | `*_acqfix*`, merged into the primary files; previous kept as `.pre_acqfix` | 2026-09-10 | 147 min | ER lead robust (10/10 partitions); PR and HER2 partition-dependent |

## Where each published item comes from

| Item | File |
|---|---|
| Table 2 | `step05b_repeated_nested_cv_primary_summary.csv`, `step05b_patient_bootstrap_primary.csv`, `step05master_*_best_model_test_result.csv` |
| Table S1 | `step07_mil_model_results.csv`, `step08_cnn_model_results.csv`, `step09_final_comparison_all_stages.csv` |
| Table S2, S8 | `step12_ebcombat_sensitivity_results.csv` |
| Table S3, Figure 6 | `step14_power_analysis.csv` |
| Table S4 | `step16_preprocessing_sensitivity.csv` |
| Figure S7, top; per-dataset values in the repository, not in the supplement | `step17_optimism_gap.csv` |
| Table S5 | `step05b_repeated_nested_cv_onesession_summary.csv` |
| Table S6, Figure S6, bottom | `step15_robustness.csv`, `step15_robustness_onesession.csv` |
| Feature blocks by name; released, not tabulated | `metadata/step04_feature_dictionary.json` |
| Table S9 | `step05b_*_agg_median*`, `step05b_*_agg_largest*` |
| Figure S3, top and middle | `step05b_crossfitted_scores_rocscores.csv` |

Result tables live in `bcbm_project/reports/tables/`, metadata in `bcbm_project/metadata/`, figures in
`bcbm_project/reports/figures/step13_publication_package/`. The release repository mirrors them under
`results/`; `scratchpad/sync_repo.py` copies by correspondence, so nothing drifts silently.

Supplementary figures S1–S20 were merged into the seven composites S1–S7 on 2026-09-12 to meet the journal's 12-page guidance for supplemental material; `tools/build_supp_composites.py` tiles them and holds the grouping, and the twenty source figures remain in the step-13 package unchanged. Old → new: S1,S2,S13,S14 → S1 | S3,S15,S16 → S2 | S4,S5,S6 → S3 | S7,S8 → S4 | S9,S10,S11 → S5 | S12,S17 → S6 | S18,S19,S20 → S7.

On the same date the supplementary tables lost two raw listings to the repository — the per-dataset radMLBench values (`results/tables/step17_optimism_gap.csv`) and the non-radiomic column names (`results/metadata/step04_feature_dictionary.json`) — and the survivors renumbered: S6→S5, S7→S6, S8→S7, S10→S8, S11→S9, S12→S10. Three figure panels whose numbers a table already carried were dropped at the same time (precision–recall curves, the multiple-instance scatter, the preprocessing plot).
