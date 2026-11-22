# Analysis protocol for the added analyses

_Public copy. Internal file references and correspondence labels were removed for publication; the design, the lock date, the reporting rules and every logged deviation are unchanged._

**Locked:** 2026-11-12, before any of the analyses below was run.
**Rule:** anything decided after this date is logged under "Deviations" at the end, with its date
and reason. Every result is reported whatever its direction. No existing result file is overwritten.
Each new run writes under its own tag and gets a new row in `docs/ANALYSES.md`.

Shared settings: `GLOBAL_SEED = 42`. Repeat *r* uses seed `42 + 1000·r`, as in step 05b. Five outer
and three inner folds. The model pool is `build_candidate_models` from step 05, unchanged, and every
preprocessing, ComBat and selection step is fitted inside the training fold. Targets are `ER_bin`,
`PR_bin` and `HER2_bin`.

---

## A1. Leakage demonstration: random versus patient-grouped splitting

**Purpose.** Measure how much splitting at the lesion or case level, instead of by patient, inflates
AUROC when everything else is held fixed. This is a methodological demonstration of bias, not a
clinical validation strategy, and the manuscript will say so.

**Pool.** The 122 train + valid patients of the primary analysis. The 17 locked-test patients stay
out, so the pool matches Table 2. Lesions or cases with a missing label for a receptor are dropped
for that receptor only.

**Arms.**

| Arm | Unit | Rows | Features | Outer and inner splitter |
|---|---|---|---|---|
| L-rand | lesion | lesions of the 122 patients | 107 lesion radiomic features (`original_*`, step 03 lesion table) | `StratifiedKFold`, shuffled |
| L-group | lesion | same | same | `StratifiedGroupKFold` by patient, at both outer and inner levels |
| C-rand | case (one T1 post-contrast volume) | cases of the 122 patients | case-level radiomic aggregates from `analysis_ready_step04_case_level.csv` (exact column list recorded at run time) | `StratifiedKFold`, shuffled |
| C-group | case | same | same | `StratifiedGroupKFold` by patient, at both levels |
| P (reference) | patient | 122 | 642-column radiomic block | **not re-run.** Read from `step05b_repeated_nested_cv_primary_summary.csv` |

ComBat batch is the field strength of the lesion's or case's own examination, mapped to the same
column the patient-level run uses.

**The only change between the arms of a pair is the splitter.** The features, model pool,
preprocessing, seeds and fold counts are identical within L-rand/L-group and within C-rand/C-group.
Comparing either pair with arm P also changes the unit and the feature representation; the text will
say that this comparison is not controlled.

**Repeats.** 10 per arm and receptor.

**Primary metric.** Mean outer-fold AUROC at the arm's own unit, as `mean_fold_auroc` in step 05b.
The summary is the mean, SD and 2.5–97.5 percentile across the 10 repeats.

**Contrast.** L-rand minus L-group, and C-rand minus C-group, paired by seed, per receptor. We report
the mean difference, its range, and the number of repeats in which the random arm is higher. The
contrast is descriptive, so no p-value is computed.

**Interpretation rule (fixed now).** Inflation is "demonstrated" for a receptor if the mean paired
difference is ≥ 0.05 AUROC and the random arm is higher in at least 9 of 10 repeats. Otherwise the
manuscript reports that splitting by lesion or case did not by itself inflate performance in this
cohort, and the leakage argument in the Discussion is weakened accordingly.

**Secondary, Taha-like arm.** A Random Forest (scikit-learn defaults, `n_estimators = 500`) on lesion
rows, with no tuning:
- (a) a random 80:20 lesion split over 100 seeds;
- (b) a patient-grouped 80:20 split over the same 100 seeds.

The distribution of test AUROC is reported for each. This shows the published protocol's style of
evaluation, not its exact run. That run cannot be reproduced, because:
1. the descriptor used 297 images and the release has 268, with no mapping between them;
2. the lesion filtering rules are not described;
3. the seed, folds and hyperparameters were not published.

**Not reproduced.** The published AUROCs (0.81, 0.75 and 0.77 for ER, PR and HER2) are quoted for
context only.

## A2. Lesion-level learning with patient-level separation

**Purpose.** Test whether keeping lesions, or cases, as separate training rows adds signal that
patient-level aggregation discards.

**Source.** The out-of-fold lesion and case probabilities from the grouped arms of A1. No extra
model fitting is needed.

**Aggregation.** The primary rule is fixed now: the patient score is the **mean** of the patient's
out-of-fold lesion (or case) probabilities within a repeat. The maximum and the largest-lesion
probability (by `mask_volume_cc`) are secondary sensitivity rules and are never promoted to primary.

**Metrics.**
1. The patient-level AUROC per repeat, summarized across the 10 repeats as in step 05b.
2. The cross-fitted patient AUROC: each patient's score averaged over repeats, with a 2,000-sample
   patient bootstrap 95% CI.
3. Lesion-level AUROC as a secondary metric, flagged as dependent on lesion burden.

**Permutation test (conditional, fixed now).** It runs only if the cross-fitted CI of the primary rule
excludes 0.50 for a receptor. It uses 1,000 permutations of patient labels (all of a patient's
lesions move together) through the reduced elastic-net pipeline of step 15 under grouped CV.
Benjamini–Hochberg correction is applied across the three receptors.

**Interpretation rule (fixed now).** Lesion-level learning "recovers signal" for a receptor only if
the cross-fitted CI excludes 0.50 and the corrected permutation p is below 0.05. In that case the
Abstract, Discussion and Conclusion will state it, and will say that patient-level aggregation was
conservative. Otherwise, reliable discrimination was not demonstrated with lesion-level learning
either.

## A3. Included versus excluded patients

**Groups.** The 139 analyzed patients versus the 26 others among the 165. The 26 are split by the
reason recorded in step 03:

| Reason | Patients |
|---|---|
| No receptor label of any kind in the released clinical sheet | 23 (22 of them have lesion segmentations) |
| Receptor labels present but no structure classified as a lesion | 3 |

These counts were found while preparing this protocol, before any comparison was run.

**Variables.**
- Age and year at the first examination.
- Dominant field strength.
- Number of examinations.
- Number of lesion masks and total lesion volume per patient. These use the same lesion definition
  (`final_mask_category == "lesion"`) for every group, before label filtering.
- ER, PR and HER2 status, where present.

Sex (all female) and manufacturer (all Siemens) are reported as constant.

**Statistics.** Mann–Whitney U for continuous variables and Fisher's exact test for categorical ones.
The standardized mean difference is the main summary, because n = 26. The comparison is descriptive
and has no correction.

**Script and output.** `src/analysis/step_20_inclusion_comparison.py`, writing
`step20_inclusion_comparison.csv` and `step20_excluded_patient_reasons.csv`.

## A4. Detectable effect at the patient level

**Reading.** The existing cross-fitted CI is read from `step05b_patient_bootstrap_primary.csv`.

**New calculation.** The minimum AUROC detectable with 80% power at α = 0.05, by Hanley–McNeil, for
the per-receptor class counts of the patients in the cross-fitted analysis, treating patients (not
pooled predictions) as the unit. It is written as new rows tagged `patient_unit` in a separate file.
`step14_power_analysis.csv` is left untouched.

**Text.** The nominal 0.95–0.98 figure is removed from the Abstract and Discussion. If it is kept in
the supplement, it is labelled as an upper bound under independence.

## A5. Attenuation of AUROC by label discordance

**Model.** An analytic simulation with Gaussian scores. A fraction ε of labels is flipped at random,
nondifferentially, for true AUROC values of 0.65, 0.75 and 0.85 and ε from 0 to 0.3. The observed
AUROC is reported, using 200,000 simulated cases per cell and seed 42.

**Discordance rates.** Only rates quoted from a source whose content has been checked.

**Output.** One supplementary figure and table.

## A6. Reproducibility files

The partition of every patient (and lesion or case) to its fold, for each repeat and arm of step 05b
and A1/A2, is written to `results/partitions/` together with the seeds. These files are produced by
re-deriving the splitters from their seeds, without refitting any model.

## E7. radMLBench: typical versus favorable split

This needs no new run. The text reports `single_split_median` beside `single_split_p95` from
`step17_optimism_gap.csv`.

## S24. Full-procedure permutation null: reporting rule

**Written 2026-11-18, after an interim look.** The run (step 24, `--part fullnull`) was started on
2026-11-15 without a written reporting rule. Interim draws were read on 2026-11-17 (16-17
permutations per receptor) and 2026-11-18 (23-24). At the second look the one-sided p values were
ER 0.84, PR 0.84 and HER2 0.042 (0 of 23 null draws at or above the observed 0.569). This rule
was written after those looks and is reported as such.

- **Count.** 50 permutations per receptor, fixed on 2026-11-15 for run time, before any interim
  comparison with the observed values. The run is neither stopped early nor extended because of
  the results. If it cannot finish, the number completed and the reason are reported.
- **Statistic.** Mean fold AUROC of `radiomics_pure` over the ten step 05b partitions, compared
  with the same quantity under permuted labels. One-sided p = (k + 1) / (N + 1), where k is the
  number of null draws at or above the observed value.
- **Multiplicity.** Holm correction across ER, PR and HER2. Raw and adjusted p are both reported.
  With N = 50 the smallest possible raw p is 1/51 = 0.020, so the smallest possible Holm-adjusted
  p is 3/51 = 0.059. No receptor can pass the corrected 0.05 threshold in this run.
- **Wording.**
  - Raw p < 0.05: "the observed AUROC was above the permutation null at the nominal level
    (p = x), but not after correction for three receptors (adjusted p = y)". It is described as
    a weak, unconfirmed signal. The text also gives the patient-bootstrap CI and the reduced
    pipeline result (HER2 p = 0.55), and makes no claim of predictive use.
  - Raw p >= 0.05: "the observed AUROC did not exceed the permutation null".
  - The main conclusion stays as written: no reproducible receptor discrimination was shown.
- **Calibration.** The null mean and SD for each receptor are reported as a check that the full
  procedure does not inflate AUROC when no signal is present.
- **Other step 24 parts.** The reduced-pipeline null over ten partitions (D1), the 20-lesion cap
  (D5/B13) and the one-sided label flips (A15/B6) are reported whatever their direction. All three
  had finished before this rule was written.

---

## Deviations

- **2026-11-12, A3, field strength.** "Dominant field strength" is computed as in step 04: the mode
  over the patient's lesion rows, or over all rows for patients with no lesion mask. The first run
  took the mode over examinations and gave 7 instead of 11 patients at 3 T. It was corrected before
  the results were interpreted, so the included group matches Table 1 (126 at 1.5 T, 11 at 3 T).
- **2026-11-12, A4.** The patient-unit Hanley-McNeil calculation reproduces step 14's out-of-fold
  rows exactly: step 14 already counted each patient once (119/116/115). The independence concern
  therefore lies in the dependence between the five fold models, which Hanley-McNeil cannot
  represent, not in pooled duplicates. Two quantities were added to make that visible: the SE
  implied by the existing patient bootstrap and the upper 95% CI bound. Class counts come from
  `step05b_crossfitted_scores_acqfix.csv`, because the primary run saved no per-patient scores;
  the totals are checked against the primary bootstrap file.
- **2026-11-12, A5.** The protocol did not fix a prevalence; 0.5 was used (ER 0.49, HER2 0.52 in
  the analysis pool). The closed form 0.5 + (A - 0.5)(1 - 2e), exact at prevalence 0.5, is reported
  beside each simulated value as a check.
- **2026-11-12, A6.** Partitions are written for the primary step 05b run (tag `primary`). For
  A1/A2 only outer folds are written, as recorded by the run; inner folds were not stored.
- **2026-11-18, step 24 full null.** The reporting rule (section S24) was written after two interim
  looks at the null draws; see that section for the interim values.

- **2026-11-22, A2, permutation trigger.** The lesion-trained cross-fitted intervals lie wholly below 0.50, so they "exclude 0.50" in the wording of the conditional permutation test above. That test was written to confirm a signal above chance (see the interpretation rule) and was not run; the below-chance estimates have an identified cause, patients with hundreds of lesions (step 24, lesion arms capped at 20 per patient). Noticed in the final reading of the manuscript, after the results were known.
