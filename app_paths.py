"""Centralized filesystem paths and model-artifact helpers for the Streamlit app."""

from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "battery_data.csv"
METADATA_PATH = ROOT_DIR / "model_metadata.json"
PREPROCESSING_REPORT_PATH = ROOT_DIR / "preprocessing_report.json"


def artifact_path(filename: str) -> Path:
    """Return the canonical path for a persisted model/scaler artifact."""
    return ROOT_DIR / filename


def model_path(task: str, model_name: str) -> Path:
    """Return the canonical path for a persisted model artifact."""
    safe_task = task.strip().lower()
    if safe_task not in {"soh", "rul"}:
        raise ValueError("task must be 'soh' or 'rul'")

    safe_name = model_name.strip().lower().replace(" ", "_")
    return artifact_path(f"{safe_task}_{safe_name}.joblib")


def required_artifacts(task: str, model_name: str) -> list[Path]:
    """Return all artifacts required to run a prediction task."""
    prefix = task.strip().lower()
    return [METADATA_PATH, model_path(prefix, model_name), artifact_path(f"{prefix}_scaler.joblib")]


def artifacts_available(task: str, model_name: str) -> bool:
    """Check whether all required artifacts exist."""
    return all(path.is_file() for path in required_artifacts(task, model_name))


def first_existing(*paths: Path) -> Optional[Path]:
    """Return the first existing file from the supplied paths."""
    return next((path for path in paths if path.is_file()), None)
