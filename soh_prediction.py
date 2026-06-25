"""
Battery Intelligence Platform — SoH Prediction Page
======================================================
Input form for battery metrics, SoH gauge meter,
health classification, confidence score, and recommendations.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import joblib
import os
import json

from src.utils import (
    classify_risk, get_risk_color, get_risk_icon, get_maintenance_recommendations,
    calculate_confidence, RISK_CSS_CLASSES,
)
from src.model_evaluation import COLORS, _plotly_dark_layout
from src.feature_engineering import SOH_FEATURES


def _create_soh_gauge(soh_value: float) -> go.Figure:
    """Create a premium SoH gauge meter."""
    category = classify_risk(soh_value)
    color = get_risk_color(category)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=soh_value,
        number=dict(
            font=dict(size=52, color=COLORS["text"], family="Inter"),
            suffix="%",
        ),
        title=dict(
            text="State of Health",
            font=dict(size=16, color=COLORS["muted"], family="Inter"),
        ),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickwidth=1,
                tickcolor=COLORS["muted"],
                dtick=20,
                tickfont=dict(size=11, color=COLORS["muted"]),
            ),
            bar=dict(color=color, thickness=0.75),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                {"range": [0, 60], "color": "rgba(248,113,113,0.12)"},
                {"range": [60, 75], "color": "rgba(251,191,36,0.12)"},
                {"range": [75, 90], "color": "rgba(96,165,250,0.12)"},
                {"range": [90, 100], "color": "rgba(52,211,153,0.12)"},
            ],
            threshold=dict(
                line=dict(color=COLORS["text"], width=3),
                thickness=0.8,
                value=soh_value,
            ),
        ),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=COLORS["text"]),
        height=300,
        margin=dict(l=30, r=30, t=60, b=20),
    )
    return fig


def _load_model_and_scaler():
    """Load the best SoH model and scaler from saved artifacts."""
    models_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
    )
    metadata_path = os.path.join(models_dir, "model_metadata.json")

    if not os.path.exists(metadata_path):
        return None, None, None

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    best_name = metadata.get("soh_best_model", "XGBoost")
    safe_name = best_name.lower().replace(" ", "_")
    model_path = os.path.join(models_dir, f"soh_{safe_name}.joblib")
    scaler_path = os.path.join(models_dir, "soh_scaler.joblib")

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler, best_name
    return None, None, None


def render():
    """Render the SoH Prediction page."""

    st.markdown(
        '<p class="hero-title">🔋 Battery Health Prediction</p>'
        '<p class="hero-subtitle">Predict the State of Health (SoH) of lithium-ion EV batteries</p>',
        unsafe_allow_html=True,
    )

    # Load model
    model, scaler, model_name = _load_model_and_scaler()
    if model is None:
        st.warning(
            "⚠️ Models not trained yet. Please run the training pipeline first.\n\n"
            "```bash\ncd battery-intelligence-platform\npython -c \"from src.model_training import train_all_models; train_all_models()\"\n```"
        )
        return

    st.markdown(f"*Using model: **{model_name}***", unsafe_allow_html=True)
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # --- Input Form ---
    col_input, col_result = st.columns([1, 1.3])

    with col_input:
        st.markdown('<div class="section-header">📝 Battery Parameters</div>', unsafe_allow_html=True)
        st.markdown('<p class="section-subheader">Enter current battery metrics for prediction</p>', unsafe_allow_html=True)

        with st.form("soh_prediction_form", clear_on_submit=False):
            c1, c2 = st.columns(2)
            with c1:
                voltage = st.number_input("Voltage (V)", min_value=2.5, max_value=4.5, value=3.65, step=0.01, format="%.2f")
                current = st.number_input("Current (A)", min_value=-5.0, max_value=5.0, value=-2.0, step=0.1, format="%.1f")
                temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=28.0, step=0.5)
            with c2:
                resistance = st.number_input("Internal Resistance (Ω)", min_value=0.01, max_value=0.5, value=0.035, step=0.001, format="%.4f")
                capacity = st.number_input("Capacity (Ah)", min_value=0.5, max_value=3.0, value=1.75, step=0.01, format="%.2f")
                cycles = st.number_input("Charge Cycles", min_value=1, max_value=2000, value=200, step=10)

            # Additional inputs with expander
            with st.expander("⚙️ Advanced Parameters", expanded=False):
                voltage_charge = st.number_input("Charge Voltage (V)", min_value=3.5, max_value=4.5, value=4.18, step=0.01)
                current_charge = st.number_input("Charge Current (A)", min_value=0.1, max_value=5.0, value=1.5, step=0.1)
                ambient_temp = st.number_input("Ambient Temperature (°C)", min_value=-10.0, max_value=50.0, value=23.0, step=0.5)

            submitted = st.form_submit_button("⚡ Predict Battery Health", use_container_width=True)

    with col_result:
        if submitted:
            # Build feature vector
            # We need to compute engineered features from the raw inputs
            initial_capacity = 2.0  # assumed nominal
            capacity_retention_rate = capacity / initial_capacity
            initial_resistance = 0.022
            resistance_growth_rate = (resistance - initial_resistance) / initial_resistance
            cycle_efficiency = (abs(current) * voltage) / max(current_charge * voltage_charge, 0.01)
            cycle_efficiency = min(max(cycle_efficiency, 0.5), 1.2)
            degradation_rate = max(0, (initial_capacity - capacity) / max(cycles, 1))
            optimal_low, optimal_high = 20.0, 30.0
            if temperature < optimal_low:
                temp_stress = (optimal_low - temperature) / 10.0
            elif temperature > optimal_high:
                temp_stress = (temperature - optimal_high) / 10.0
            else:
                temp_stress = 0.0
            wear_index = 0.5 * max(0, resistance_growth_rate) + 0.5 * max(0, 1 - capacity_retention_rate)
            voltage_drop = voltage_charge - voltage
            cumulative_temp_stress = temp_stress * cycles * 0.01

            # Feature vector matching SOH_FEATURES order
            feature_values = {
                "cycle": cycles,
                "voltage_measured": voltage,
                "current_measured": current,
                "temperature_measured": temperature,
                "voltage_charge": voltage_charge,
                "current_charge": current_charge,
                "capacity": capacity,
                "internal_resistance": resistance,
                "ambient_temperature": ambient_temp,
                "capacity_retention_rate": capacity_retention_rate,
                "resistance_growth_rate": resistance_growth_rate,
                "cycle_efficiency": cycle_efficiency,
                "degradation_rate": degradation_rate,
                "temperature_stress_score": temp_stress,
                "battery_wear_index": wear_index,
                "avg_charge_temperature": temperature,
                "avg_discharge_temperature": temperature * 0.98,
                "voltage_drop": voltage_drop,
                "cumulative_temp_stress": cumulative_temp_stress,
            }

            X_input = pd.DataFrame([feature_values])[SOH_FEATURES]
            X_scaled = pd.DataFrame(scaler.transform(X_input), columns=SOH_FEATURES)

            # Predict
            soh_pred = float(model.predict(X_scaled)[0])
            soh_pred = max(0, min(100, soh_pred))
            category = classify_risk(soh_pred)
            color = get_risk_color(category)
            icon = get_risk_icon(category)
            confidence = calculate_confidence(model, X_scaled, "soh")

            # --- Display Results ---
            st.markdown('<div class="section-header">📊 Prediction Results</div>', unsafe_allow_html=True)

            # Gauge
            fig_gauge = _create_soh_gauge(soh_pred)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Category & Confidence
            r1, r2 = st.columns(2)
            with r1:
                css_class = RISK_CSS_CLASSES.get(category, "badge-good")
                st.markdown(
                    f'<div style="text-align:center">'
                    f'<span class="{css_class}">{icon} {category}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with r2:
                st.markdown(
                    f'<div style="text-align:center">'
                    f'<span style="color:{COLORS["accent"]};font-size:1.1rem;font-weight:600">'
                    f'🎯 Confidence: {confidence:.1f}%</span></div>',
                    unsafe_allow_html=True,
                )

            # Store prediction in session state for history
            if "soh_history" not in st.session_state:
                st.session_state.soh_history = []
            st.session_state.soh_history.append({
                "SoH": round(soh_pred, 1),
                "Category": category,
                "Confidence": confidence,
                "Cycles": cycles,
            })

        else:
            st.markdown('<div class="section-header">📊 Prediction Results</div>', unsafe_allow_html=True)
            st.info("👈 Enter battery parameters and click **Predict Battery Health** to see results.")

    # --- Recommendations Section ---
    if submitted:
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">💡 Smart Maintenance Recommendations</div>', unsafe_allow_html=True)

        recommendations = get_maintenance_recommendations(
            soh=soh_pred, temperature=temperature, resistance=resistance
        )

        for rec in recommendations:
            priority_color = {
                "info": COLORS["accent"],
                "warning": COLORS["warning"],
                "critical": COLORS["danger"],
            }.get(rec["priority"], COLORS["accent"])

            st.markdown(
                f'<div class="recommendation-card" style="border-left-color:{priority_color}">'
                f'{rec["icon"]} {rec["message"]}</div>',
                unsafe_allow_html=True,
            )

    # --- Prediction History ---
    if "soh_history" in st.session_state and len(st.session_state.soh_history) > 0:
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📜 Prediction History</div>', unsafe_allow_html=True)
        hist_df = pd.DataFrame(st.session_state.soh_history[-10:])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
