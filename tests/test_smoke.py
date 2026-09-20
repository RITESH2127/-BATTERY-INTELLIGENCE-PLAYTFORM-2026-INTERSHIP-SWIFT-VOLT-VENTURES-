from pathlib import Path

import pandas as pd

from feature_engineering import engineer_features, SOH_FEATURES, RUL_FEATURES
from utils import classify_risk, generate_fleet_data

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_and_feature_engineering():
    df = pd.read_csv(ROOT / "battery_data.csv")
    engineered = engineer_features(df)
    assert len(engineered) == len(df)
    assert set(SOH_FEATURES).issubset(engineered.columns)
    assert set(RUL_FEATURES).issubset(engineered.columns)


def test_risk_and_fleet_helpers():
    assert classify_risk(95) == "Excellent"
    assert classify_risk(50) == "Critical"

    fleet = generate_fleet_data(25, seed=42)
    assert len(fleet) == 25
    assert fleet["Battery ID"].is_unique
