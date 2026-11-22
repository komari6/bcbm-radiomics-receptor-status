# Analyses on record

Every analysis that produced a number in the paper is listed here with the command that produced it,
its outputs and its headline result. All values are from the run of 13–14 November 2026 on the
corrected lesion set (1,522 lesions in 140 patients; train/tuning/test 103/20/17 patients; 120, 117
and 116 patients with an ER, PR and HER2 label in the cross-validation pool of 123), except steps
24 and 13c (run afterwards on the same data) and steps 00–02, 10, 17 and 22, whose inputs the
correction did not touch. Values are ER / PR / HER2 unless stated.

**Rule: do not re-run an analysis to recover a number. Read it back from the file named below.**
A sensitivity run never writes to the primary outputs: `step_05b` and `step_15` take `--tag`, and
`step_04` appends a suffix (`_onesession`, `_agg_median`, `_agg_largest`).

## Primary analyses

| Analysis | Command | Key outputs (`results/`) | Result |
|---|---|---|---|
| Feature engineering (642 radiomic, 21 acquisition, 79 burden/study columns) | `python src/data/step_04_feature_engineering.py --force` | `metadata/step04_feature_dictionary.json`, `metadata/step04_patient_splits.csv` | 140 patients; split 103/20/17 |
| Scanner-confounding audit | inside step 04 | `metadata/step04_scanner_confounding_audit.json`, `step04_scanner_radiomics_shift_screening.csv` | 6 of 107 features differ by field strength at the patient level (BH); 105 of 107 when lesions are wrongly treated as independent |
| Repeated nested CV, primary endpoint (Table 2) | `python src/modeling/step_05b_repeated_nested_cv.py --jobs 10` | `tables/step05b_repeated_nested_cv_primary_{summary,per_repeat}.csv`, `step05b_patient_bootstrap_primary.csv`, `step05b_crossfitted_scores_primary.csv` | Radiomic block 0.47 / 0.46 / 0.57; cross-fitted 0.46 (0.35–0.57), 0.40 (0.29–0.52), 0.57 (0.47–0.68); acquisition block 0.57 / 0.51 / 0.58 |
| Single locked split (locked-test column of Table 2) | `python src/modeling/step_05_tabular_modeling.py --force` | `tables/step05master_*` | Radiomic block 0.38 / 0.75 / 0.52 on 16–17 patients; maximum 0.78 (HER2, acquisition block) |
| Deep models (S1 Table A) | `step_07` (MIL), `step_08` (CNN), `step_09` (fusion; run after step 05) | `step07_mil_model_results.csv`, `step08_cnn_model_results.csv`, `step09_hybrid_model_results.csv`, `step09_final_comparison_all_stages.csv` | Locked test: attention-MIL 0.62 / 0.76 / 0.68, CNN 0.54 / 0.49 / 0.50, tuning-selected step-09 variant 0.63 / 0.55 / 0.56 |
| Harmonization sensitivity and pairwise tests (S1 Table B) | `python src/analysis/step_12_combat_eb_and_statistical_comparison.py` | `step12_ebcombat_sensitivity_results.csv`, `step12_internal_pairwise_model_comparison.csv` | Locked-test AUROC moves by up to 0.27 with the harmonization mode; one contrast (entered twice) has BH-adjusted bootstrap p = 0.045, DeLong 0.14 |
| Learning curve and power (Figs 5–6, S1 Table C) | `python src/analysis/step_14_learning_curve_power.py` | `step14_learning_curve.csv`, `step14_power_analysis.csv` | No sustained rise with training size; locked test power 0.23–0.26 for AUROC 0.70 |
| Label permutation and seed stability (Fig 7, S1 Table F) | `python src/analysis/step_15_robustness_null.py --n-permutations 10000` (and `--tag onesession` on the one-examination table) | `step15_robustness.csv`, `step15_robustness_onesession.csv` | Nominal p 0.55 / 0.17 / 0.56 (one examination 0.81 / 0.45 / 0.51); 25-seed means 0.54 / 0.51 / 0.49 |
| Preprocessing sensitivity (S1 Table D) | `conda run -n bcbm_radiomics python src/analysis/step_16_preprocessing_sensitivity.py` | `step16_preprocessing_sensitivity.csv` | 0.44–0.58 across five extraction settings |
| radMLBench optimism (Fig 8) | `python src/analysis/step_17_radmlbench_generalization.py --max-datasets 18` | `step17_optimism_gap.csv` | Median split-to-split SD 0.11; favorable (95th-percentile) split gap median 0.17 |
| OpenBTAI proxy (S1 File) | `step_10`, `step_11` | `step11_openbtai_external_proxy_results.csv` | 7 / 7 / 20 evaluable patients, AUROC 0.33 / 0.33 / 0.52; not interpretable |
| Feature transfer to UCSF-BMSR (S1 Fig O) | `conda run -n bcbm_radiomics python src/analysis/step_18_external_feature_transfer.py` | `step18_external_feature_transfer.csv` | Features shifted in 93% of columns before and 51% after location-scale alignment |

## Sensitivity analyses

| Question | Command | Outputs | Result |
|---|---|---|---|
| One examination per patient (S1 Table E) | `step_04 --one-session --reuse-split-from <primary table>`, then `step_05b --input …_onesession.csv --feature-dict …_onesession.json --tag onesession` | `*_onesession*` | Radiomic 0.51 / 0.56 / 0.51; acquisition 0.48 / 0.50 / 0.54 |
| Median aggregation (S1 Table I) | `step_04 --aggregation median --reuse-split-from <primary table>`, then `step_05b … --variants radiomics_pure --tag agg_median` | `*_agg_median*` | 0.44 / 0.53 / 0.54 |
| Largest lesion only (S1 Table I) | as above with `--aggregation largest` / `--tag agg_largest` | `*_agg_largest*` | 0.49 / 0.54 / 0.56 |

## Analyses under the protocol (`docs/PROTOCOL_added_analyses.md`)

| Analysis | Command | Outputs | Result |
|---|---|---|---|
| A1/A2 Unit of analysis (Table 3, S1 Table M) | `python src/analysis/step_19_unit_of_analysis.py --repeats 10 --rf-seeds 100 --jobs 10 --tag unitdemo` | `step19_unitdemo_*` | Lesions split at random 0.77 / 0.78 / 0.78, by patient 0.49 / 0.56 / 0.61; cases by patient, scored per patient 0.48 / 0.43 / 0.53 |
| A3 Patients not analyzed (S1 Table K) | `python src/analysis/step_20_inclusion_comparison.py` | `step20_inclusion_comparison.csv`, `step20_excluded_patient_reasons.csv` | 25 not analyzed: 23 without any receptor label, 2 without a lesion |
| A4 Detectable effect | `python src/analysis/step_21_detectable_effect.py` | `step21_detectable_effect.csv` | Minimum detectable AUROC 0.65–0.66; bootstrap/Hanley–McNeil SE ratio 1.02–1.04 |
| A5 Label error (S1 Table L, Fig P) | `python src/analysis/step_22_label_noise_simulation.py` | `step22_label_noise_attenuation.csv` | Matches 0.5 + (A − 0.5)(1 − 2ε) within 0.002 |
| A6 Fold assignments | `python src/analysis/step_23_partitions.py` | `partitions/` | Every patient held out exactly once per repetition |

## Added after the protocol (step 24; S1 Table O)

| Analysis | Command | Outputs | Result |
|---|---|---|---|
| Full-procedure permutation null | `python src/analysis/step_24_r1_supplementary.py --part fullnull --n-perm 50 --jobs 22` | `step24_full_procedure_null{,_draws}.csv` | Observed 0.47 / 0.46 / 0.57; null mean 0.51 / 0.51 / 0.51; one-sided p 0.84 / 0.78 / 0.22, Holm 1.00 / 1.00 / 0.65. Reporting rule (protocol section S24) written after two interim looks, recorded there |
| Reduced pipeline on the ten partitions | `--part reduced --n-perm 1000` | `step24_reduced_partitions_null{,_draws}.csv` | 0.53 / 0.52 / 0.49 (partitions 0.43–0.60); p 0.26 / 0.31 / 0.55 |
| Lesion arms capped at 20 per patient | `--part capped` | `step24_capped20_{per_repeat,summary,contrast}.csv` | Grouped-arm patient AUROC 0.51 / 0.46 / 0.54 (0.42 / 0.37 / 0.43 uncapped); random arm still higher by 0.04–0.11 |
| One-sided label error | `--part flips` | `step24_label_flips_one_sided.csv` | True 0.75 at class-conditional error 0.10 / 0.20 / 0.30: 0.73 / 0.71 / 0.69, against 0.70 / 0.65 / 0.60 symmetric |

## Where each published item comes from

| Item | File |
|---|---|
| Table 2 | `step05b_repeated_nested_cv_primary_summary.csv`, `step05b_patient_bootstrap_primary.csv`, `step05master_*_best_model_test_result.csv` |
| Table 3, S1 Table M | `step19_unitdemo_{summary,contrast,patient_crossfitted,rf_single_split}.csv` |
| S1 Table A, S1 Fig N | `step07_mil_model_results.csv`, `step08_cnn_model_results.csv`, `step09_hybrid_model_results.csv`, `step09_final_comparison_all_stages.csv` |
| S1 Table B | `step12_ebcombat_sensitivity_results.csv`, `step12_internal_pairwise_model_comparison.csv` |
| S1 Table C, Fig 6 | `step14_power_analysis.csv` |
| S1 Table D | `step16_preprocessing_sensitivity.csv` |
| S1 Table E | `step05b_repeated_nested_cv_onesession_summary.csv` |
| S1 Table F, Fig 7 | `step15_robustness.csv`, `step15_robustness_onesession.csv` |
| S1 Table I | `step05b_*_agg_median*`, `step05b_*_agg_largest*` |
| S1 Table J, S1 Fig G | `step05b_crossfitted_scores_primary.csv` |
| S1 Table K | `step20_inclusion_comparison.csv` |
| S1 Table L, S1 Fig P | `step22_label_noise_attenuation.csv` |
| S1 Table O | `step24_*` |
| Fig 8 | `step17_optimism_gap.csv` |
| Feature blocks by name | `metadata/step04_feature_dictionary.json` |
