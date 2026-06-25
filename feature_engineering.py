"""
Battery Intelligence Platform — Feature Engineering
=====================================================
Derives advanced features from raw battery cycling data to improve
ML model performance for SoH and RUL prediction.
"""

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features for battery health prediction.

    Parameters
    ----------
    df : pd.DataFrame
        Raw battery cycling data with columns: battery_id, cycle, voltage_measured,
        current_measured, temperature_measured, voltage_charge, current_charge,
        capacity, internal_resistance, ambient_temperature, soh, rul.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional engineered features.
    """
    df = df.copy()
    df = df.sort_values(["battery_id", "cycle"]).reset_index(drop=True)

    # Group by battery for per-battery calculations
    grouped = df.groupby("battery_id")

    # 1. Capacity Retention Rate — current capacity / initial capacity
    initial_capacity = grouped["capacity"].transform("first")
    df["capacity_retention_rate"] = df["capacity"] / initial_capacity

    # 2. Internal Resistance Growth Rate — resistance change from baseline
    initial_resistance = grouped["internal_resistance"].transform("first")
    df["resistance_growth_rate"] = (df["internal_resistance"] - initial_resistance) / initial_resistance

    # 3. Cycle Efficiency — ratio of energy output metrics
    # Using |current_measured * voltage_measured| / (current_charge * voltage_charge)
    charge_power = df["current_charge"].abs() * df["voltage_charge"].abs()
    discharge_power = df["current_measured"].abs() * df["voltage_measured"].abs()
    df["cycle_efficiency"] = np.where(
        charge_power > 0,
        discharge_power / charge_power,
        0.95  # default fallback
    )
    df["cycle_efficiency"] = df["cycle_efficiency"].clip(0.5, 1.2)

    # 4. Degradation Rate — rolling capacity loss per cycle (window=20)
    df["degradation_rate"] = grouped["capacity"].transform(
        lambda x: -x.diff().rolling(window=20, min_periods=1).mean()
    )
    df["degradation_rate"] = df["degradation_rate"].fillna(0).clip(-0.01, 0.05)

    # 5. Temperature Stress Score — deviation from optimal range (20-30°C)
    optimal_low, optimal_high = 20.0, 30.0
    df["temperature_stress_score"] = np.where(
        df["temperature_measured"] < optimal_low,
        (optimal_low - df["temperature_measured"]) / 10.0,
        np.where(
            df["temperature_measured"] > optimal_high,
            (df["temperature_measured"] - optimal_high) / 10.0,
            0.0
        )
    )

    # 6. Battery Wear Index — composite of resistance growth + capacity fade
    capacity_fade = 1.0 - df["capacity_retention_rate"]
    df["battery_wear_index"] = (
        0.5 * df["resistance_growth_rate"].clip(0, 10) +
        0.5 * capacity_fade.clip(0, 1)
    )

    # 7. Average Charging Temperature — rolling mean (window=30)
    df["avg_charge_temperature"] = grouped["temperature_measured"].transform(
        lambda x: x.rolling(window=30, min_periods=1).mean()
    )

    # 8. Average Discharge Temperature — rolling mean with lag
    df["avg_discharge_temperature"] = grouped["temperature_measured"].transform(
        lambda x: x.rolling(window=50, min_periods=1).mean()
    )

    # 9. Voltage drop — difference between charge voltage and measured voltage
    df["voltage_drop"] = df["voltage_charge"] - df["voltage_measured"]

    # 10. Cumulative temperature stress
    df["cumulative_temp_stress"] = grouped["temperature_stress_score"].cumsum()

    # Fill any remaining NaN values
    df = df.ffill().bfill().fillna(0)

    return df


# Features used for model training
SOH_FEATURES = [
    "cycle", "voltage_measured", "current_measured", "temperature_measured",
    "voltage_charge", "current_charge", "capacity", "internal_resistance",
    "ambient_temperature", "capacity_retention_rate", "resistance_growth_rate",
    "cycle_efficiency", "degradation_rate", "temperature_stress_score",
    "battery_wear_index", "avg_charge_temperature", "avg_discharge_temperature",
    "voltage_drop", "cumulative_temp_stress",
]

RUL_FEATURES = [
    "cycle", "voltage_measured", "current_measured", "temperature_measured",
    "voltage_charge", "current_charge", "capacity", "internal_resistance",
    "ambient_temperature", "capacity_retention_rate", "resistance_growth_rate",
    "cycle_efficiency", "degradation_rate", "temperature_stress_score",
    "battery_wear_index", "avg_charge_temperature", "avg_discharge_temperature",
    "voltage_drop", "cumulative_temp_stress",
]

SOH_TARGET = "soh"
RUL_TARGET = "rul"
