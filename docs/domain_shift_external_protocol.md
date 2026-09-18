# External domain-shift / feature-transfer protocol (Stage 4)

**Status:** protocol only — not executed automatically (requires large external downloads and CaPTk preprocessing; CPU, GPU optional for CaPTk's deep skull-strip). This is the heaviest improvement and is documented for manual execution.

## Why (and what it can / cannot show)
No public cohort pairs brain-metastasis MRI with breast-primary **receptor status**, so a true external *receptor* validation is impossible. However, two public, segmented, T1-post-contrast brain-metastasis datasets contain a **breast-primary subset**:

- **UCSF-BMSR** (412 patients; expert metastasis masks) — Rudie et al., *Radiol Artif Intell* 2024. https://pubs.rsna.org/doi/full/10.1148/ryai.230126
- **Stanford BrainMetShare** (105 patients) — via the BraTS-METS / Stanford release.

They label **primary tumor type** (breast identifiable) but **not ER/PR/HER2**. They therefore support a **domain-shift / feature-transfer** analysis (do BCBM-derived radiomic features distribute and transfer to an independent breast-BM cohort?), **not** receptor validation. This directly tests the generalizability concern raised in the manuscript.

## Required preprocessing parity (critical)
Radiomic features are not comparable across preprocessing. Process **both** the BCBM images and the external breast-subset images through the **identical** pipeline before any comparison:

1. Use **CaPTk BraTS preprocessing** (https://cbica.github.io/CaPTk/preprocessing_brats.html): reorient → rigid registration to **SRI-24** atlas (1 mm isotropic) → (deep) skull-strip. Note: CaPTk applies N4 only *temporarily* for registration; our BCBM `image_ss_n4` has N4 in the final image — so for parity, regenerate BCBM features from the **same** convention (either both with or both without final N4).
2. Extract the **same 107 IBSI-aligned PyRadiomics features** with one fixed settings file (binWidth, resampling, normalization) for both cohorts.

## Analyses (CPU)
1. **Feature-distribution shift:** per feature, two-sample test (Kolmogorov–Smirnov) BCBM vs external; report the fraction of features that shift significantly, and a PCA/UMAP overlay of the two cohorts.
2. **ComBat transfer test:** fit ComBat on BCBM, apply to external; report residual shift.
3. **Model transfer (no labels needed for the receptor target):** apply the BCBM-trained pipeline to the external breast subset and inspect the predicted-score distribution; if any external receptor labels become available later, compute AUROC once (apply-once).
4. **Optional primary-type sanity check:** train breast-vs-other on the external set's primary-type labels to confirm features carry *some* signal (positive control), contrasting with the receptor null.

## Expected contribution
Demonstrates whether radiomic features transfer across institutions/scanners at all — strengthening the manuscript's generalizability argument with public data, without requiring receptor labels.

## Effort / dependencies
- Downloads: UCSF-BMSR (TCIA), Stanford BrainMetShare (tens of GB).
- Tools: CaPTk (Docker recommended) or equivalent (ANTs registration to SRI-24 + HD-BET skull-strip), PyRadiomics.
- Compute: CPU sufficient; CaPTk deep skull-strip/tumor-seg can use GPU but is optional.
- A skeleton extractor can reuse `src/analysis/step_16_preprocessing_sensitivity.py` (same PyRadiomics extraction) pointed at the external NIfTI paths.

## Honest caveat
Even fully executed, this remains **feature-transfer / domain-shift**, not external **receptor** validation — the field-level data gap stands and is reported as such in the manuscript Discussion.
