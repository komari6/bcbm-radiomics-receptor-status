"""
BCBM Radio-Genomics Pipeline Runner
====================================
Runs steps 00–18 in order. Stops on any non-zero exit code unless --continue-on-error is set.

Usage examples
--------------
  python run_pipeline.py                        # full pipeline
  python run_pipeline.py --from-step 5          # resume from step 05
  python run_pipeline.py --only-step 8          # run one step
  python run_pipeline.py --skip 6 8             # skip steps
  python run_pipeline.py --force-step 8         # re-run step 8 even if cached
  python run_pipeline.py --no-cache             # disable cache for all steps
  python run_pipeline.py --external             # run steps 10 and 11 only
  python run_pipeline.py --reports-only         # run step 13 only
  python run_pipeline.py --dry-run              # print plan without running
  python run_pipeline.py --from-step 5 --skip 6 --no-cache  # combined
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    (0,  "src/data/step_00_build_master_file.py",
         "Build internal master dataset from source Excel files"),
    (1,  "src/data/step_01_project_initialization.py",
         "Initialize project folder structure and copy raw data"),
    (2,  "src/data/step_02_data_audit.py",
         "Audit identifiers and resolve NIfTI file paths"),
    (3,  "src/data/step_03_segmentation_harmonization.py",
         "Classify mask descriptors and filter to lesion-only table"),
    (4,  "src/data/step_04_feature_engineering.py",
         "Patient-level aggregation, scanner audit, and train/valid/test splits"),
    (5,  "src/modeling/step_05_tabular_modeling.py",
         "Tabular radiomics: nested CV + fold-safe ComBat + stability selection"),
    (6,  "src/modeling/step_06_image_baseline_modeling.py",
         "Shallow image baseline: histogram-PCA features (benchmark only)"),
    (7,  "src/modeling/step_07_mil_radiomics.py",
         "Radiomics Attention-MIL on lesion-level bags"),
    (8,  "src/modeling/step_08_cnn_image.py",
         "2.5D CNN-MIL on raw MRI slices (uses CUDA if available, else CPU)"),
    (9,  "src/modeling/step_09_hybrid_fusion.py",
         "Hybrid fusion: radiomics MIL + CNN image streams"),
    (10, "src/external/step_10_build_openbtai_external_proxy_dataset.py",
         "Build OpenBTAI external proxy dataset"),
    (11, "src/external/step_11_openbtai_external_proxy_validation.py",
         "External proxy validation on OpenBTAI cohort"),
    (12, "src/analysis/step_12_combat_eb_and_statistical_comparison.py",
         "EB-ComBat sensitivity analysis and pairwise statistical comparison"),
    (13, "src/reporting/step_13_figures_and_reports.py",
         "Publication figures, tables, and final package"),
    (14, "src/analysis/step_14_learning_curve_power.py",
         "Learning-curve and statistical-power analysis (sample-size adequacy)"),
    (15, "src/analysis/step_15_robustness_null.py",
         "Robustness: label-permutation null, seed stability, calibration (CPU)"),
    (16, "src/analysis/step_16_preprocessing_sensitivity.py",
         "Radiomics preprocessing-sensitivity (needs pyradiomics; CPU, slow)"),
    (17, "src/analysis/step_17_radmlbench_generalization.py",
         "Evaluation-optimism generalization across radMLBench datasets (needs radMLBench; CPU)"),
    (18, "src/analysis/step_18_external_feature_transfer.py",
         "External feature-transfer/domain-shift vs UCSF-BMSR breast subset (needs pyradiomics + external data; CPU)"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BCBM pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--from-step",   type=int, default=0,   metavar="N",
                   help="Start from step N (skip all earlier steps)")
    p.add_argument("--only-step",   type=int, default=None, metavar="N",
                   help="Run only step N and exit")
    p.add_argument("--skip",        type=int, nargs="+", default=[], metavar="N",
                   help="Skip these step numbers")
    p.add_argument("--force-step",  type=int, nargs="+", default=[], metavar="N",
                   help="Force re-run of these steps even if cache is valid")
    p.add_argument("--no-cache",    action="store_true",
                   help="Disable cache for all steps")
    p.add_argument("--external",    action="store_true",
                   help="Run external pipeline only (steps 10 and 11)")
    p.add_argument("--reports-only", action="store_true",
                   help="Run reporting step only (step 13)")
    p.add_argument("--dry-run",     action="store_true",
                   help="Print the execution plan without running any step")
    p.add_argument("--continue-on-error", action="store_true",
                   help="Log failures and continue instead of halting the pipeline")
    return p.parse_args()


def build_env_flags(args: argparse.Namespace, step_num: int) -> list[str]:
    """Return extra CLI flags to pass to a step script based on runner flags."""
    flags: list[str] = []
    if args.no_cache:
        flags.append("--no-cache")
    if step_num in args.force_step:
        flags.append("--force")
    return flags


def run_step(num: int, script: str, label: str, extra_flags: list[str],
             dry_run: bool, continue_on_error: bool) -> bool:
    script_path = ROOT / script
    if not script_path.exists():
        print(f"[MISSING]  Step {num:02d} — {script} not found, skipping")
        return True

    cmd = [sys.executable, str(script_path)] + extra_flags
    print(f"\n{'='*70}")
    print(f"  Step {num:02d}: {label}")
    print(f"  Script : {script}")
    if extra_flags:
        print(f"  Flags  : {' '.join(extra_flags)}")
    print(f"{'='*70}")

    if dry_run:
        print(f"  [DRY-RUN] Would execute: {' '.join(cmd)}")
        return True

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n[FAILED]  Step {num:02d} exited with code {result.returncode}  "
              f"(elapsed {elapsed:.1f}s)")
        if not continue_on_error:
            sys.exit(result.returncode)
        return False

    print(f"[OK]  Step {num:02d} done  ({elapsed:.1f}s)")
    return True


def write_run_manifest(steps_run: list[dict], failed: list[int], dry_run: bool) -> None:
    metadata_dir = ROOT / "bcbm_project" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_at": datetime.utcnow().isoformat(),
        "dry_run": dry_run,
        "steps_run": steps_run,
        "failed_steps": failed,
        "success": len(failed) == 0,
    }
    path = metadata_dir / "pipeline_run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    # Determine which steps to run
    if args.external:
        target_steps = {10, 11}
    elif args.reports_only:
        target_steps = {13}
    elif args.only_step is not None:
        target_steps = {args.only_step}
    else:
        target_steps = None  # all

    skip_set = set(args.skip)
    steps_run: list[dict] = []
    failed: list[int] = []

    print("\nBCBM Pipeline Runner")
    print(f"  Start  : {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  DRY-RUN: {'YES' if args.dry_run else 'no'}")
    print(f"  From   : step {args.from_step:02d}" if not args.external and not args.reports_only else "")

    for num, script, label in STEPS:
        if target_steps is not None and num not in target_steps:
            continue
        if num < args.from_step and target_steps is None:
            continue
        if num in skip_set:
            print(f"[SKIP]  Step {num:02d}: {label}")
            continue

        extra_flags = build_env_flags(args, num)
        ok = run_step(num, script, label, extra_flags, args.dry_run, args.continue_on_error)
        steps_run.append({"step": num, "script": script, "success": ok})
        if not ok:
            failed.append(num)

    if not args.dry_run:
        write_run_manifest(steps_run, failed, args.dry_run)

    if failed:
        print(f"\n[DONE]  Pipeline finished with {len(failed)} failed step(s): {failed}")
        sys.exit(1)
    else:
        print(f"\n[DONE]  Pipeline complete — all {len(steps_run)} step(s) succeeded.")


if __name__ == "__main__":
    main()
