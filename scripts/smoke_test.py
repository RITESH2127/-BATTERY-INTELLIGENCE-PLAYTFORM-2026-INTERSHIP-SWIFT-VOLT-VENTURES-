"""Fast repository smoke test for CI and local verification."""

from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import engineer_features, SOH_FEATURES, RUL_FEATURES
from utils import classify_risk, generate_fleet_data
from soh_prediction import _load_model_and_scaler as load_soh_artifacts
from rul_prediction import _load_model_and_scaler as load_rul_artifacts
from explainability_dashboard import _load_models_and_data as load_xai_artifacts
from app_paths import DATA_PATH, METADATA_PATH, model_path, artifact_path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    required = [
        ROOT / "app.py",
        ROOT / "battery_data.csv",
        ROOT / "requirements.txt",
        ROOT / "model_metadata.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")

    expected_models = [
        model_path("soh", "Gradient Boosting"),
        model_path("rul", "LightGBM"),
        artifact_path("soh_scaler.joblib"),
        artifact_path("rul_scaler.joblib"),
    ]
    missing_models = [str(p) for p in expected_models if not p.is_file()]
    if missing_models:
        raise SystemExit(f"Missing required model artifacts: {missing_models}")

    assert DATA_PATH.is_file()
    assert METADATA_PATH.is_file()

    df = pd.read_csv(DATA_PATH)
    engineered = engineer_features(df)

    assert len(engineered) == len(df)
    assert set(SOH_FEATURES).issubset(engineered.columns)
    assert set(RUL_FEATURES).issubset(engineered.columns)
    values = engineered[SOH_FEATURES + RUL_FEATURES].replace([np.inf, -np.inf], np.nan)
    assert values.notna().all().all()

    fleet = generate_fleet_data(25, seed=42)
    assert len(fleet) == 25
    assert set(fleet["Status"]).issubset({"Excellent", "Good", "Warning", "Critical"})
    assert classify_risk(95) == "Excellent"
    assert classify_risk(50) == "Critical"

    soh_model, soh_scaler, soh_name = load_soh_artifacts()
    rul_model, rul_scaler, rul_name = load_rul_artifacts()
    xai_artifacts = load_xai_artifacts()
    assert soh_model is not None and soh_scaler is not None and soh_name
    assert rul_model is not None and rul_scaler is not None and rul_name
    assert xai_artifacts is not None

    print("Smoke test passed: dataset, paths, model artifacts, prediction loaders, and XAI loader are healthy.")


if __name__ == "__main__":
    main()
