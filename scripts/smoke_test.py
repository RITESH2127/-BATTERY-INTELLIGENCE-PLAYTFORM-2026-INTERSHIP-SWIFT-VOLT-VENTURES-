"""Fast repository smoke test for CI and local verification."""

from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import engineer_features, SOH_FEATURES, RUL_FEATURES
from utils import classify_risk, generate_fleet_data

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

    df = pd.read_csv(ROOT / "battery_data.csv")
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

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
