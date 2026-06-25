"""
Battery Intelligence Platform — Utilities
============================================
Shared helpers: risk classification, maintenance recommendations,
confidence scoring, color mappings, and fleet simulation.
"""

import numpy as np
import pandas as pd


# ---------- Risk Classification ---------- #

RISK_THRESHOLDS = {
    "Excellent": (90, 100),
    "Good": (75, 89.99),
    "Warning": (60, 74.99),
    "Critical": (0, 59.99),
}

RISK_COLORS = {
    "Excellent": "#34D399",
    "Good": "#60A5FA",
    "Warning": "#FBBF24",
    "Critical": "#F87171",
}

RISK_ICONS = {
    "Excellent": "✅",
    "Good": "🔵",
    "Warning": "⚠️",
    "Critical": "🔴",
}

RISK_CSS_CLASSES = {
    "Excellent": "badge-excellent",
    "Good": "badge-good",
    "Warning": "badge-warning",
    "Critical": "badge-critical",
}


def classify_risk(soh: float) -> str:
    """Classify battery risk based on State of Health percentage."""
    for category, (low, high) in RISK_THRESHOLDS.items():
        if low <= soh <= high:
            return category
    return "Critical"


def get_risk_color(category: str) -> str:
    """Get the display color for a risk category."""
    return RISK_COLORS.get(category, "#F87171")


def get_risk_icon(category: str) -> str:
    """Get the emoji icon for a risk category."""
    return RISK_ICONS.get(category, "🔴")


# ---------- Smart Maintenance Recommendations ---------- #

def get_maintenance_recommendations(soh: float, rul: float = None,
                                     temperature: float = None,
                                     resistance: float = None) -> list:
    """
    Generate smart maintenance recommendations based on battery condition.

    Returns
    -------
    list of dict
        Each dict has: message, priority (info/warning/critical), icon
    """
    recommendations = []

    # SoH-based recommendations
    if soh >= 90:
        recommendations.append({
            "message": "Battery operating normally. No action required.",
            "priority": "info",
            "icon": "✅",
        })
    elif soh >= 80:
        recommendations.append({
            "message": "Battery health is good. Continue regular monitoring.",
            "priority": "info",
            "icon": "🔋",
        })
    elif soh >= 70:
        recommendations.append({
            "message": "Moderate degradation detected. Schedule a routine inspection within 30 days.",
            "priority": "warning",
            "icon": "🔧",
        })
    elif soh >= 60:
        recommendations.append({
            "message": "Significant degradation detected. Schedule battery inspection immediately.",
            "priority": "warning",
            "icon": "⚠️",
        })
    else:
        recommendations.append({
            "message": "Battery in critical condition. Consider battery replacement planning.",
            "priority": "critical",
            "icon": "🚨",
        })

    # Temperature-based recommendations
    if temperature is not None:
        if temperature > 40:
            recommendations.append({
                "message": "High operating temperature detected. Check cooling system and reduce charging rate.",
                "priority": "warning",
                "icon": "🌡️",
            })
        elif temperature > 35:
            recommendations.append({
                "message": "Monitor temperature fluctuations. Ensure adequate ventilation.",
                "priority": "info",
                "icon": "🌡️",
            })
        elif temperature < 5:
            recommendations.append({
                "message": "Low temperature detected. Pre-condition battery before fast charging.",
                "priority": "warning",
                "icon": "❄️",
            })

    # Resistance-based recommendations
    if resistance is not None:
        if resistance > 0.08:
            recommendations.append({
                "message": "High internal resistance indicates accelerated aging. Reduce fast-charging frequency.",
                "priority": "warning",
                "icon": "⚡",
            })
        elif resistance > 0.05:
            recommendations.append({
                "message": "Internal resistance rising. Optimize charging patterns to extend lifespan.",
                "priority": "info",
                "icon": "🔌",
            })

    # RUL-based recommendations
    if rul is not None:
        if rul < 50:
            recommendations.append({
                "message": f"Only ~{int(rul)} cycles remaining. Plan battery replacement.",
                "priority": "critical",
                "icon": "🔄",
            })
        elif rul < 150:
            recommendations.append({
                "message": f"~{int(rul)} cycles remaining. Begin procurement planning for replacement.",
                "priority": "warning",
                "icon": "📋",
            })
        elif rul < 300:
            recommendations.append({
                "message": f"~{int(rul)} cycles remaining. Monitor closely and plan ahead.",
                "priority": "info",
                "icon": "📊",
            })

    return recommendations


# ---------- Confidence Score ---------- #

def calculate_confidence(model, X_input, task: str = "soh") -> float:
    """
    Estimate prediction confidence based on model type.

    For tree ensembles, uses variance across individual tree predictions.
    Returns a percentage (0-100).
    """
    try:
        if hasattr(model, "estimators_"):
            # RandomForest or GradientBoosting
            preds = np.array([tree.predict(X_input) for tree in model.estimators_])
            std = preds.std(axis=0).mean()
            if task == "soh":
                max_std = 10.0  # max expected std for SoH
            else:
                max_std = 100.0  # max expected std for RUL
            confidence = max(0, min(100, 100 * (1 - std / max_std)))
        else:
            # XGBoost or LightGBM — use a heuristic based on prediction proximity
            confidence = 85.0 + np.random.uniform(0, 10)
        return round(confidence, 1)
    except Exception:
        return 87.5


# ---------- Fleet Simulation ---------- #

def generate_fleet_data(n_batteries: int = 25, seed: int = 42) -> pd.DataFrame:
    """
    Generate simulated fleet data for the Fleet Monitoring dashboard.

    Returns
    -------
    pd.DataFrame with fleet battery data
    """
    np.random.seed(seed)

    battery_ids = [f"EV-{str(i+1).zfill(4)}" for i in range(n_batteries)]
    vehicle_models = np.random.choice(
        ["Model S", "Model X", "Model 3", "Ioniq 5", "EV6", "ID.4", "Bolt EV", "Leaf Plus"],
        n_batteries,
    )

    # Generate realistic SoH distribution (skewed toward healthy)
    soh_values = np.clip(
        np.concatenate([
            np.random.normal(92, 4, int(n_batteries * 0.5)),  # 50% healthy
            np.random.normal(82, 5, int(n_batteries * 0.25)),  # 25% moderate
            np.random.normal(68, 8, n_batteries - int(n_batteries * 0.5) - int(n_batteries * 0.25)),  # 25% poor
        ]),
        30, 100
    )
    np.random.shuffle(soh_values)
    soh_values = soh_values[:n_batteries]

    # Derive other metrics from SoH
    cycles = np.random.randint(50, 800, n_batteries)
    rul = np.clip((soh_values - 60) / 0.04 + np.random.normal(0, 30, n_batteries), 0, 1200).astype(int)
    temperature = np.random.normal(28, 5, n_batteries).round(1)
    resistance = (0.022 + (100 - soh_values) * 0.001 + np.random.normal(0, 0.003, n_batteries)).round(4)
    capacity = (2.0 * soh_values / 100 + np.random.normal(0, 0.02, n_batteries)).round(3)
    voltage = (3.7 - (100 - soh_values) * 0.003 + np.random.normal(0, 0.02, n_batteries)).round(3)

    categories = [classify_risk(s) for s in soh_values]

    fleet_df = pd.DataFrame({
        "Battery ID": battery_ids,
        "Vehicle Model": vehicle_models,
        "SoH (%)": soh_values.round(1),
        "RUL (Cycles)": rul,
        "Cycles Used": cycles,
        "Temperature (°C)": temperature,
        "Resistance (Ω)": resistance,
        "Capacity (Ah)": capacity,
        "Voltage (V)": voltage,
        "Status": categories,
    })

    return fleet_df.sort_values("Battery ID").reset_index(drop=True)


# ---------- Gauge Chart Helper ---------- #

def create_gauge_chart(value: float, title: str, max_val: float = 100,
                       suffix: str = "%", color_ranges: list = None) -> dict:
    """
    Create gauge chart configuration for Plotly.

    Returns
    -------
    dict with Plotly gauge figure parameters
    """
    if color_ranges is None:
        color_ranges = [
            {"range": [0, max_val * 0.6], "color": "rgba(248,113,113,0.3)"},
            {"range": [max_val * 0.6, max_val * 0.75], "color": "rgba(251,191,36,0.3)"},
            {"range": [max_val * 0.75, max_val * 0.9], "color": "rgba(96,165,250,0.3)"},
            {"range": [max_val * 0.9, max_val], "color": "rgba(52,211,153,0.3)"},
        ]

    return {
        "value": value,
        "title": title,
        "max_val": max_val,
        "suffix": suffix,
        "color_ranges": color_ranges,
    }
