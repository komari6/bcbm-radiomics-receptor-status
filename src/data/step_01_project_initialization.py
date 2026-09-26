from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# =========================================================
# 1) DEFAULT SETTINGS
# =========================================================

_SCRIPT_DIR = Path(".").resolve()  # project root (cwd set by run_pipeline.py)

DEFAULT_INPUT_XLSX_NAME = "BCBM_Full_Data.xlsx"
DEFAULT_RAW_DATASET_DIR_NAME = "BCBM-RadioGenomics_Images_Masks_Dec2024"
DEFAULT_PROJECT_NAME = "bcbm_radiogenomics"
DEFAULT_PROJECT_ROOT = Path("./bcbm_project").resolve()
DEFAULT_GLOBAL_SEED = 42
DEFAULT_COPY_XLSX = True
DEFAULT_COPY_RAW_FOLDER = True
DEFAULT_OVERWRITE_EXISTING = False


# =========================================================
# 2) CONFIG
# =========================================================

@dataclass
class ProjectConfig:
    project_name: str = DEFAULT_PROJECT_NAME
    project_root: str = str(DEFAULT_PROJECT_ROOT)

    raw_data_dir: str = ""
    interim_data_dir: str = ""
    processed_data_dir: str = ""
    metadata_dir: str = ""
    logs_dir: str = ""
    checkpoints_dir: str = ""
    reports_dir: str = ""
    figures_dir: str = ""
    tables_dir: str = ""
    models_dir: str = ""
    archive_dir: str = ""

    script_dir: str = str(_SCRIPT_DIR)
    input_xlsx_original: str = ""
    input_xlsx_project_copy: str = ""
    raw_dataset_source_dir: str = ""
    raw_dataset_project_copy_dir: str = ""

    copy_xlsx_to_project: bool = DEFAULT_COPY_XLSX
    copy_raw_folder_to_project: bool = DEFAULT_COPY_RAW_FOLDER
    overwrite_existing: bool = DEFAULT_OVERWRITE_EXISTING
    global_seed: int = DEFAULT_GLOBAL_SEED
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    patient_id_column_candidates: list[str] = field(default_factory=lambda: [
        "PatientBase", "Patient_ID", "PatientID", "patient_id", "patient_base",
    ])
    lesion_name_column_candidates: list[str] = field(default_factory=lambda: [
        "Segmentation_Name", "segmentation_name", "mask_name", "MaskName",
    ])
    label_column_candidates: dict[str, list[str]] = field(default_factory=lambda: {
        "ER": ["ER", "er", "ER_status", "er_status"],
        "PR": ["PR", "pr", "PR_status", "pr_status"],
        "HER2": ["HER2", "her2", "HER2_status", "her2_status"],
    })


# =========================================================
# 3) STATE MANAGER
# =========================================================

class StateManager:
    def __init__(self, state_path: Path):
        self.state_path = state_path
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if self.state_path.exists():
            with open(self.state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "project_initialized": False,
            "steps_completed": [],
            "artifacts": {},
            "notes": [],
            "last_updated": None,
        }

    def save(self) -> None:
        self.state["last_updated"] = datetime.utcnow().isoformat()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    def mark_step_done(self, step_name: str, artifacts: Optional[Dict[str, str]] = None) -> None:
        if step_name not in self.state["steps_completed"]:
            self.state["steps_completed"].append(step_name)
        if artifacts:
            self.state["artifacts"].update(artifacts)
        if step_name == "step_01_project_initialization":
            self.state["project_initialized"] = True
        self.save()

    def add_note(self, note: str) -> None:
        self.state.setdefault("notes", []).append({
            "time": datetime.utcnow().isoformat(),
            "note": note,
        })
        self.save()

    def is_step_done(self, step_name: str) -> bool:
        return step_name in self.state.get("steps_completed", [])


# =========================================================
# 4) LOGGING / HELPERS
# =========================================================


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)



def setup_logger(log_file: Path, verbose: bool = True) -> logging.Logger:
    logger = logging.getLogger("bcbm_project")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if verbose:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    logger.info("=" * 90)
    logger.info("NEW SESSION STARTED at %s", datetime.utcnow().isoformat())
    logger.info("=" * 90)
    return logger



def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)



def detect_excel_engine(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        return "openpyxl"
    if suffix == ".xls":
        return "xlrd"
    raise ValueError(f"Unsupported Excel file extension: {suffix}")



def safe_read_excel(file_path: str, sheet_name=0) -> pd.DataFrame:
    engine = detect_excel_engine(file_path)
    xl = pd.ExcelFile(file_path, engine=engine)
    sheet = xl.sheet_names[0] if len(xl.sheet_names) == 1 else sheet_name
    return pd.read_excel(file_path, sheet_name=sheet, engine=engine)



def create_initial_data_profile(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "missing_per_column": {k: int(v) for k, v in df.isna().sum().to_dict().items()},
        "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 3),
    }



def write_manifest(manifest_path: Path, config: ProjectConfig, dataframe_info: Optional[Dict[str, Any]] = None) -> None:
    manifest = {
        "project_name": config.project_name,
        "created_at": config.created_at,
        "project_root": config.project_root,
        "script_dir": config.script_dir,
        "input_xlsx_original": config.input_xlsx_original,
        "input_xlsx_project_copy": config.input_xlsx_project_copy,
        "raw_dataset_source_dir": config.raw_dataset_source_dir,
        "raw_dataset_project_copy_dir": config.raw_dataset_project_copy_dir,
        "copy_xlsx_to_project": config.copy_xlsx_to_project,
        "copy_raw_folder_to_project": config.copy_raw_folder_to_project,
        "overwrite_existing": config.overwrite_existing,
        "global_seed": config.global_seed,
        "dataframe_info": dataframe_info or {},
    }
    save_json(manifest, manifest_path)



def copy_file_if_needed(src: Path, dst: Path, overwrite: bool, logger: logging.Logger) -> Path:
    if not src.exists():
        raise FileNotFoundError(f"Required source file not found: {src}")
    ensure_dir(dst.parent)
    if dst.exists():
        if overwrite:
            shutil.copy2(src, dst)
            logger.info("Overwrote file copy: %s -> %s", src, dst)
        else:
            logger.info("Project file copy already exists, keeping existing file: %s", dst)
    else:
        shutil.copy2(src, dst)
        logger.info("Copied file to project raw data: %s -> %s", src, dst)
    return dst



def copy_directory_if_needed(src: Path, dst: Path, overwrite: bool, logger: logging.Logger) -> Path:
    if not src.exists():
        raise FileNotFoundError(f"Required source folder not found: {src}")
    if not src.is_dir():
        raise NotADirectoryError(f"Expected a directory but found: {src}")

    ensure_dir(dst.parent)

    if dst.exists():
        if overwrite:
            shutil.rmtree(dst)
            shutil.copytree(src, dst)
            logger.info("Overwrote dataset folder copy: %s -> %s", src, dst)
        else:
            logger.info("Project raw dataset folder already exists, keeping existing folder: %s", dst)
    else:
        shutil.copytree(src, dst)
        logger.info("Copied raw dataset folder to project: %s -> %s", src, dst)
    return dst


# =========================================================
# 5) PROJECT INITIALIZATION
# =========================================================


def build_project_config(
    project_root: Path,
    script_dir: Path,
    input_xlsx_name: str,
    raw_dataset_dir_name: str,
    copy_xlsx_to_project: bool,
    copy_raw_folder_to_project: bool,
    overwrite_existing: bool,
    global_seed: int,
) -> ProjectConfig:
    cfg = ProjectConfig()
    cfg.project_root = str(project_root.resolve())
    cfg.script_dir = str(script_dir.resolve())

    cfg.raw_data_dir = str(project_root / "data" / "raw")
    cfg.interim_data_dir = str(project_root / "data" / "interim")
    cfg.processed_data_dir = str(project_root / "data" / "processed")
    cfg.metadata_dir = str(project_root / "metadata")
    cfg.logs_dir = str(project_root / "logs")
    cfg.checkpoints_dir = str(project_root / "checkpoints")
    cfg.reports_dir = str(project_root / "reports")
    cfg.figures_dir = str(project_root / "reports" / "figures")
    cfg.tables_dir = str(project_root / "reports" / "tables")
    cfg.models_dir = str(project_root / "models")
    cfg.archive_dir = ""  # not used

    input_xlsx_original = script_dir / input_xlsx_name
    raw_dataset_source_dir = script_dir / raw_dataset_dir_name

    cfg.input_xlsx_original = str(input_xlsx_original)
    cfg.input_xlsx_project_copy = str(project_root / "data" / "raw" / input_xlsx_original.name)
    cfg.raw_dataset_source_dir = str(raw_dataset_source_dir)
    cfg.raw_dataset_project_copy_dir = str(project_root / "data" / "raw" / raw_dataset_source_dir.name)

    cfg.copy_xlsx_to_project = copy_xlsx_to_project
    cfg.copy_raw_folder_to_project = copy_raw_folder_to_project
    cfg.overwrite_existing = overwrite_existing
    cfg.global_seed = global_seed
    return cfg



def initialize_project_structure(cfg: ProjectConfig) -> None:
    root = Path(cfg.project_root)
    dirs = [
        # data
        Path(cfg.raw_data_dir),
        root / "data" / "interim",
        root / "data" / "processed",
        root / "data" / "external_ready",
        root / "data" / "external_raw",
        # metadata / logs / checkpoints
        Path(cfg.metadata_dir),
        Path(cfg.logs_dir),
        Path(cfg.checkpoints_dir),
        # reports
        Path(cfg.reports_dir),
        Path(cfg.figures_dir),
        Path(cfg.tables_dir),
        root / "reports" / "packages",
        # cache (per sub-type)
        root / "cache" / "masks",
        root / "cache" / "image_features",
        root / "cache" / "cnn_embeddings",
        root / "cache" / "radiomics_bags",
        root / "cache" / "tabular_preprocessing",
        root / "cache" / "external_mapping",
        # models (per step)
        root / "models" / "step05_tabular",
        root / "models" / "step07_mil",
        root / "models" / "step08_cnn",
        root / "models" / "step09_hybrid",
        root / "models" / "step11_external_refit",
    ]
    for d in dirs:
        ensure_dir(d)



def validate_sources_exist(cfg: ProjectConfig) -> None:
    xlsx_path = Path(cfg.input_xlsx_original)
    xlsx_dst  = Path(cfg.input_xlsx_project_copy)
    raw_folder_path = Path(cfg.raw_dataset_source_dir)

    missing = []
    # If step_00 already wrote BCBM_Full_Data.xlsx to data/raw/, skip source check.
    if cfg.copy_xlsx_to_project and not xlsx_path.exists() and not xlsx_dst.exists():
        missing.append(str(xlsx_path))
    if cfg.copy_raw_folder_to_project and not raw_folder_path.exists():
        missing.append(str(raw_folder_path))

    if missing:
        raise FileNotFoundError(
            "Missing required source items:\n- " + "\n- ".join(missing)
        )


# =========================================================
# 6) CLI
# =========================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Initialize BCBM project structure, copy BCBM_Full_Data.xlsx and the "
            "BCBM-RadioGenomics_Images_Masks_Dec2024 folder into bcbm_project/data/raw, "
            "then generate initial metadata and validation outputs."
        )
    )
    parser.add_argument("--project-root", type=str, default=str(DEFAULT_PROJECT_ROOT), help="Target project root directory.")
    parser.add_argument("--script-dir", type=str, default=str(_SCRIPT_DIR), help="Directory containing the source Excel file and raw data folder.")
    parser.add_argument("--input-xlsx-name", type=str, default=DEFAULT_INPUT_XLSX_NAME, help="Excel filename expected inside script-dir.")
    parser.add_argument("--raw-dataset-dir-name", type=str, default=DEFAULT_RAW_DATASET_DIR_NAME, help="Raw dataset folder name expected inside script-dir.")
    parser.add_argument("--seed", type=int, default=DEFAULT_GLOBAL_SEED, help="Global random seed.")
    parser.add_argument("--no-copy-xlsx", action="store_true", help="Do not copy the Excel file into project raw data.")
    parser.add_argument("--no-copy-raw-folder", action="store_true", help="Do not copy the raw image/mask folder into project raw data.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing copied file/folder inside project raw data.")
    parser.add_argument("--force-rerun", action="store_true", help="Run Stage 01 again even if state says it is completed.")
    parser.add_argument("--quiet", action="store_true", help="Disable console logging and keep only file logging.")
    return parser.parse_args()


# =========================================================
# 7) MAIN
# =========================================================


def main() -> None:
    args = parse_args()
    step_name = "step_01_project_initialization"

    project_root = Path(args.project_root).resolve()
    script_dir = Path(args.script_dir).resolve()

    cfg = build_project_config(
        project_root=project_root,
        script_dir=script_dir,
        input_xlsx_name=args.input_xlsx_name,
        raw_dataset_dir_name=args.raw_dataset_dir_name,
        copy_xlsx_to_project=not args.no_copy_xlsx,
        copy_raw_folder_to_project=not args.no_copy_raw_folder,
        overwrite_existing=args.overwrite,
        global_seed=args.seed,
    )

    initialize_project_structure(cfg)

    state_path = Path(cfg.checkpoints_dir) / "pipeline_state.json"
    state = StateManager(state_path)

    log_file = Path(cfg.logs_dir) / f"{step_name}.log"
    logger = setup_logger(log_file=log_file, verbose=not args.quiet)

    logger.info("Starting project initialization")
    logger.info("Project root: %s", cfg.project_root)
    logger.info("Script dir: %s", cfg.script_dir)
    logger.info("Excel source expected at: %s", cfg.input_xlsx_original)
    logger.info("Raw dataset folder source expected at: %s", cfg.raw_dataset_source_dir)

    if state.is_step_done(step_name) and not args.force_rerun:
        logger.info("%s already completed. Use --force-rerun to run again.", step_name)
        return

    validate_sources_exist(cfg)

    set_global_seed(cfg.global_seed)
    logger.info("Global seed set to %d", cfg.global_seed)

    config_path = Path(cfg.metadata_dir) / "project_config.json"
    save_json(asdict(cfg), config_path)
    logger.info("Saved project config: %s", config_path)

    copied_artifacts: Dict[str, str] = {}

    if cfg.copy_xlsx_to_project:
        dst_xlsx = Path(cfg.input_xlsx_project_copy)
        if dst_xlsx.exists() and not cfg.overwrite_existing:
            # Step 00 already wrote BCBM_Full_Data.xlsx here — skip redundant copy.
            logger.info("BCBM_Full_Data.xlsx already at destination (written by Step 00), skipping copy.")
            copied_xlsx = dst_xlsx
        else:
            src_xlsx = Path(cfg.input_xlsx_original)
            if not src_xlsx.exists():
                # Fall back: if not in cwd, it must already be at destination
                logger.warning("Source xlsx not found at %s — assuming already at destination.", src_xlsx)
                copied_xlsx = dst_xlsx
            else:
                copied_xlsx = copy_file_if_needed(
                    src=src_xlsx,
                    dst=dst_xlsx,
                    overwrite=cfg.overwrite_existing,
                    logger=logger,
                )
        copied_artifacts["input_xlsx_project_copy"] = str(copied_xlsx)
    else:
        copied_xlsx = Path(cfg.input_xlsx_original)
        logger.info("Skipping Excel copy as requested.")

    if cfg.copy_raw_folder_to_project:
        copied_raw_dir = copy_directory_if_needed(
            src=Path(cfg.raw_dataset_source_dir),
            dst=Path(cfg.raw_dataset_project_copy_dir),
            overwrite=cfg.overwrite_existing,
            logger=logger,
        )
        copied_artifacts["raw_dataset_project_copy_dir"] = str(copied_raw_dir)
    else:
        logger.info("Skipping raw folder copy as requested.")

    logger.info("Reading dataset for initial validation from: %s", copied_xlsx)
    df = safe_read_excel(str(copied_xlsx))
    logger.info("Dataset loaded successfully with shape: %s", df.shape)

    profile = create_initial_data_profile(df)
    profile_path = Path(cfg.metadata_dir) / "initial_data_profile.json"
    save_json(profile, profile_path)
    logger.info("Saved initial data profile: %s", profile_path)

    preview_path = Path(cfg.interim_data_dir) / "dataset_head.csv"
    df.head(20).to_csv(preview_path, index=False, encoding="utf-8-sig")
    logger.info("Saved dataset preview: %s", preview_path)

    columns_path = Path(cfg.metadata_dir) / "columns.txt"
    with open(columns_path, "w", encoding="utf-8") as f:
        for col in df.columns:
            f.write(str(col) + "\n")
    logger.info("Saved column list: %s", columns_path)

    manifest_path = Path(cfg.metadata_dir) / "manifest.json"
    write_manifest(
        manifest_path,
        cfg,
        dataframe_info={
            "n_rows": profile["n_rows"],
            "n_columns": profile["n_columns"],
        },
    )
    logger.info("Saved manifest: %s", manifest_path)

    stage_summary = {
        "step_name": step_name,
        "completed_at": datetime.utcnow().isoformat(),
        "project_root": cfg.project_root,
        "script_dir": cfg.script_dir,
        "copied_excel": cfg.copy_xlsx_to_project,
        "copied_raw_folder": cfg.copy_raw_folder_to_project,
        "overwrite_existing": cfg.overwrite_existing,
        "input_xlsx_original": cfg.input_xlsx_original,
        "input_xlsx_project_copy": cfg.input_xlsx_project_copy,
        "raw_dataset_source_dir": cfg.raw_dataset_source_dir,
        "raw_dataset_project_copy_dir": cfg.raw_dataset_project_copy_dir,
        "dataset_shape": {
            "n_rows": int(df.shape[0]),
            "n_columns": int(df.shape[1]),
        },
        "artifacts": {
            **copied_artifacts,
            "project_config_json": str(config_path),
            "initial_data_profile_json": str(profile_path),
            "dataset_preview_csv": str(preview_path),
            "columns_txt": str(columns_path),
            "manifest_json": str(manifest_path),
            "log_file": str(log_file),
        },
    }

    summary_path = Path(cfg.metadata_dir) / "step01_summary.json"
    save_json(stage_summary, summary_path)
    logger.info("Saved stage summary: %s", summary_path)

    state.mark_step_done(
        step_name,
        artifacts={
            **copied_artifacts,
            "project_config_json": str(config_path),
            "initial_data_profile_json": str(profile_path),
            "dataset_preview_csv": str(preview_path),
            "columns_txt": str(columns_path),
            "manifest_json": str(manifest_path),
            "step01_summary_json": str(summary_path),
            "step01_log": str(log_file),
        },
    )

    logger.info("Project initialization completed successfully.")


if __name__ == "__main__":
    main()
