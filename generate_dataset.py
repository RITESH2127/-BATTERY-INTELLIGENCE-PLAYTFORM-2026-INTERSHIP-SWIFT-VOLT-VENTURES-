"""
Battery Intelligence Platform — Synthetic Dataset Generator
============================================================
Generates realistic lithium-ion battery cycling data modeled on
NASA PCoE Battery Aging Dataset characteristics.

Simulates 4 batteries (B0005, B0006, B0007, B0018) with different
aging profiles, capacity fade curves, and degradation patterns.
"""

import numpy as np
import pandas as pd
import os

# Reproducibility
SEED = 42
np.random.seed(SEED)

# Battery profiles: (battery_id, total_cycles, initial_capacity, fade_rate, knee_point_cycle)
BATTERY_PROFILES = [
    ("B0005", 616, 1.86, 0.0012, 480),
    ("B0006", 616, 2.04, 0.0010, 500),
    ("B0007", 616, 1.89, 0.0014, 450),
    ("B0018", 900, 1.85, 0.0008, 700),
]

# End-of-life threshold (fraction of initial capacity)
EOL_THRESHOLD = 0.70


def _capacity_fade_curve(cycles: np.ndarray, initial_cap: float,
                         fade_rate: float, knee_point: int) -> np.ndarray:
    """
    Non-linear capacity fade model:
      - Slow linear fade initially
      - Accelerated exponential fade after knee point
      - Random walk noise
    """
    capacity = np.zeros_like(cycles, dtype=float)
    for i, c in enumerate(cycles):
        if c <= knee_point:
            # Linear region with slight square-root curvature
            fade = fade_rate * c + 0.00002 * c ** 1.2
        else:
            # Accelerated region after knee point
            linear_fade = fade_rate * knee_point + 0.00002 * knee_point ** 1.2
            extra = (c - knee_point)
            fade = linear_fade + fade_rate * 2.5 * extra + 0.00005 * extra ** 1.5
        capacity[i] = initial_cap - fade
    # Add realistic noise (±0.5%)
    noise = np.random.normal(0, initial_cap * 0.003, len(cycles))
    capacity = capacity + noise
    capacity = np.clip(capacity, 0.3, initial_cap)
    return capacity


def _generate_battery_data(battery_id: str, total_cycles: int,
                           initial_capacity: float, fade_rate: float,
                           knee_point: int) -> pd.DataFrame:
    """Generate cycling data for a single battery."""
    cycles = np.arange(1, total_cycles + 1)
    n = len(cycles)

    # Capacity fade
    capacity = _capacity_fade_curve(cycles, initial_capacity, fade_rate, knee_point)

    # Voltage: slowly decreases with degradation
    nominal_voltage = 3.7
    voltage_measured = nominal_voltage - 0.0003 * cycles + np.random.normal(0, 0.02, n)
    voltage_measured = np.clip(voltage_measured, 3.0, 4.2)

    # Voltage charge: higher than measured, slight increase with aging
    voltage_charge = 4.2 - 0.0001 * cycles + np.random.normal(0, 0.015, n)
    voltage_charge = np.clip(voltage_charge, 3.8, 4.25)

    # Current measured (discharge): relatively constant with noise
    current_measured = -2.0 + np.random.normal(0, 0.1, n)

    # Current charge: relatively constant
    current_charge = 1.5 + np.random.normal(0, 0.05, n)

    # Temperature: increases slightly with aging and has seasonal-like variation
    base_temp = 24.0
    temp_trend = 0.003 * cycles  # gradual increase
    temp_seasonal = 2.0 * np.sin(2 * np.pi * cycles / 200)  # seasonal oscillation
    temp_noise = np.random.normal(0, 1.5, n)
    temperature_measured = base_temp + temp_trend + temp_seasonal + temp_noise
    temperature_measured = np.clip(temperature_measured, 18, 45)

    # Ambient temperature
    ambient_temperature = 23.0 + 2.0 * np.sin(2 * np.pi * cycles / 300) + np.random.normal(0, 1.0, n)
    ambient_temperature = np.clip(ambient_temperature, 18, 35)

    # Internal resistance: increases with aging (inverse of capacity health)
    base_resistance = 0.022  # Ohms
    resistance_trend = 0.00005 * cycles + 0.0000001 * cycles ** 1.5
    # Knee-point acceleration for resistance too
    for i in range(n):
        if cycles[i] > knee_point:
            extra = cycles[i] - knee_point
            resistance_trend[i] += 0.0002 * extra + 0.0000005 * extra ** 1.3
    resistance_noise = np.random.normal(0, 0.001, n)
    internal_resistance = base_resistance + resistance_trend + resistance_noise
    internal_resistance = np.clip(internal_resistance, 0.015, 0.2)

    # Compute State of Health (SoH) as percentage
    soh = (capacity / initial_capacity) * 100
    soh = np.clip(soh, 0, 100)

    # Compute Remaining Useful Life (cycles until capacity < EOL_THRESHOLD * initial)
    eol_capacity = EOL_THRESHOLD * initial_capacity
    # Find the cycle where capacity first drops below EOL
    eol_cycle = total_cycles  # default if never reaches EOL
    cap_clean = _capacity_fade_curve(cycles, initial_capacity, fade_rate, knee_point)
    for i in range(n):
        if cap_clean[i] < eol_capacity:
            eol_cycle = cycles[i]
            break
    rul = np.maximum(eol_cycle - cycles, 0).astype(float)

    df = pd.DataFrame({
        "battery_id": battery_id,
        "cycle": cycles,
        "voltage_measured": np.round(voltage_measured, 4),
        "current_measured": np.round(current_measured, 4),
        "temperature_measured": np.round(temperature_measured, 2),
        "voltage_charge": np.round(voltage_charge, 4),
        "current_charge": np.round(current_charge, 4),
        "capacity": np.round(capacity, 4),
        "internal_resistance": np.round(internal_resistance, 5),
        "ambient_temperature": np.round(ambient_temperature, 2),
        "soh": np.round(soh, 2),
        "rul": rul.astype(int),
    })

    return df


def generate_battery_data(output_dir: str = None) -> pd.DataFrame:
    """
    Generate the full synthetic battery dataset.

    Parameters
    ----------
    output_dir : str, optional
        Directory to save the CSV. Defaults to 'data/' relative to project root.

    Returns
    -------
    pd.DataFrame
        Combined dataset for all batteries.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

    os.makedirs(output_dir, exist_ok=True)

    frames = []
    for battery_id, total_cycles, init_cap, fade, knee in BATTERY_PROFILES:
        df = _generate_battery_data(battery_id, total_cycles, init_cap, fade, knee)
        frames.append(df)
        print(f"  ✓ Generated {len(df)} cycles for {battery_id}")

    combined = pd.concat(frames, ignore_index=True)
    filepath = os.path.join(output_dir, "battery_data.csv")
    combined.to_csv(filepath, index=False)
    print(f"\n✅ Dataset saved: {filepath} ({len(combined)} rows)")

    return combined


if __name__ == "__main__":
    generate_battery_data()
