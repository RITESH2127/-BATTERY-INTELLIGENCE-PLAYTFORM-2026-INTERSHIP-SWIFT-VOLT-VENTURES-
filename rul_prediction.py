"""
Battery Intelligence Platform — RUL Prediction Page
======================================================
Predict Remaining Useful Life (cycles), display gauge,
degradation forecast curve, and maintenance recommendations.
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
from src.feature_engineering import RUL_FEATURES


def _create_rul_gauge(rul_value: float, max_rul: float = 1000) -> go.Figure:
    """Create a premium RUL gauge meter."""
    # Determine color based on RUL
    if rul_value > 500:
        bar_color = "#34D399"
    elif rul_value > 200:
        bar_color = "#60A5FA"
    elif rul_value > 80:
        bar_color = "#FBBF24"
    else:
        bar_color = "#F87171"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rul_value,
        number=dict(
            font=dict(size=48, color=COLORS["text"], family="Inter"),
            suffix=" cycles",
        ),
        title=dict(
            text="Remaining Useful Life",
            font=dict(size=16, color=COLORS["muted"], family="Inter"),
        ),
        gauge=dict(
            axis=dict(
                range=[0, max_rul],
                tickwidth=1,
                tickcolor=COLORS["muted"],
                tickfont=dict(size=11, color=COLORS["muted"]),
            ),
            bar=dict(color=bar_color, thickness=0.75),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                {"range": [0, max_rul * 0.1], "color": "rgba(248,113,113,0.12)"},
                {"range": [max_rul * 0.1, max_rul * 0.3], "color": "rgba(251,191,36,0.12)"},
                {"range": [max_rul * 0.3, max_rul * 0.6], "color": "rgba(96,165,250,0.12)"},
                {"range": [max_rul * 0.6, max_rul], "color": "rgba(52,211,153,0.12)"},
            ],
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=COLORS["text"]),
        height=290,
        margin=dict(l=30, r=30, t=60, b=20),
    )
    return fig


def _create_degradation_forecast(current_soh: float, rul: float,
                                  current_cycle: int) -> go.Figure:
    """Create a degradation forecast curve."""
    # Simulate future degradation
    future_cycles = np.arange(0, int(rul * 1.3) + 50)
    total_cycles = current_cycle + future_cycles

    # Exponential decay model for capacity
    decay_rate = 0.0005 + (100 - current_soh) * 0.00002
    forecast_soh = current_soh * np.exp(-decay_rate * future_cycles)
    forecast_soh = np.clip(forecast_soh, 0, 100)

    # EOL threshold
    eol_threshold = 70.0

    fig = go.Figure()

    # Forecast line
    fig.add_trace(go.Scatter(
        x=total_cycles,
        y=forecast_soh,
        mode="lines",
        line=dict(color=COLORS["accent"], width=3, shape="spline"),
        fill="tozeroy",
        fillcolor="rgba(0,209,255,0.05)",
        name="Forecast SoH",
        hovertemplate="Cycle: %{x:.0f}<br>SoH: %{y:.1f}%<extra></extra>",
    ))

    # EOL threshold line
    fig.add_hline(
        y=eol_threshold,
        line_dash="dash",
        line_color=COLORS["danger"],
        annotation_text=f"End-of-Life ({eol_threshold}%)",
        annotation_position="top right",
        annotation_font_color=COLORS["danger"],
        annotation_font_size=12,
    )

    # Current position marker
    fig.add_trace(go.Scatter(
        x=[current_cycle],
        y=[current_soh],
        mode="markers",
        marker=dict(size=14, color=COLORS["success"], symbol="diamond",
                    line=dict(width=2, color=COLORS["text"])),
        name="Current",
        hovertemplate="<b>Current Position</b><br>Cycle: %{x}<br>SoH: %{y:.1f}%<extra></extra>",
    ))

    # Predicted EOL marker
    eol_cycle = current_cycle + int(rul)
    fig.add_trace(go.Scatter(
        x=[eol_cycle],
        y=[eol_threshold],
        mode="markers",
        marker=dict(size=14, color=COLORS["danger"], symbol="x",
                    line=dict(width=2, color=COLORS["text"])),
        name="Predicted EOL",
        hovertemplate=f"<b>Predicted EOL</b><br>Cycle: {eol_cycle}<br>SoH: {eol_threshold}%<extra></extra>",
    ))

    fig = _plotly_dark_layout(fig, "Battery Degradation Forecast")
    fig.update_layout(
        xaxis_title="Cycle Number",
        yaxis_title="State of Health (%)",
        yaxis=dict(range=[40, 105]),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _load_model_and_scaler():
    """Load the best RUL model and scaler."""
    models_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
    )
    metadata_path = os.path.join(models_dir, "model_metadata.json")

    if not os.path.exists(metadata_path):
        return None, None, None

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    best_name = metadata.get("rul_best_model", "XGBoost")
    safe_name = best_name.lower().replace(" ", "_")
    model_path = os.path.join(models_dir, f"rul_{safe_name}.joblib")
    scaler_path = os.path.join(models_dir, "rul_scaler.joblib")

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler, best_name
    return None, None, None


def render():
    """Render the RUL Prediction page."""

    st.markdown(
        '<p class="hero-title">🔄 Remaining Useful Life Prediction</p>'
        '<p class="hero-subtitle">Estimate the remaining charge cycles before end-of-life</p>',
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

    st.markdown(f"*Using model: **{model_name}***")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # --- Input Form ---
    col_input, col_result = st.columns([1, 1.3])

    with col_input:
        st.markdown('<div class="section-header">📝 Current Battery Metrics</div>', unsafe_allow_html=True)
        st.markdown('<p class="section-subheader">Enter the current operating parameters</p>', unsafe_allow_html=True)

        with st.form("rul_prediction_form", clear_on_submit=False):
            c1, c2 = st.columns(2)
            with c1:
                voltage = st.number_input("Voltage (V)", min_value=2.5, max_value=4.5, value=3.55, step=0.01, format="%.2f")
                current = st.number_input("Current (A)", min_value=-5.0, max_value=5.0, value=-2.0, step=0.1)
                temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=30.0, step=0.5)
                voltage_charge = st.number_input("Charge Voltage (V)", min_value=3.5, max_value=4.5, value=4.15, step=0.01)
            with c2:
                resistance = st.number_input("Internal Resistance (Ω)", min_value=0.01, max_value=0.5, value=0.045, step=0.001, format="%.4f")
                capacity = st.number_input("Capacity (Ah)", min_value=0.5, max_value=3.0, value=1.55, step=0.01, format="%.2f")
                cycles = st.number_input("Charge Cycles", min_value=1, max_value=2000, value=350, step=10)
                current_charge = st.number_input("Charge Current (A)", min_value=0.1, max_value=5.0, value=1.5, step=0.1)
                ambient_temp = st.number_input("Ambient Temp (°C)", min_value=-10.0, max_value=50.0, value=24.0, step=0.5)

            submitted = st.form_submit_button("🔄 Predict Remaining Life", use_container_width=True)

    with col_result:
        if submitted:
            # Build feature vector (same engineering as SoH page)
            initial_capacity = 2.0
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

            X_input = pd.DataFrame([feature_values])[RUL_FEATURES]
            X_scaled = pd.DataFrame(scaler.transform(X_input), columns=RUL_FEATURES)

            # Predict
            rul_pred = float(model.predict(X_scaled)[0])
            rul_pred = max(0, rul_pred)
            confidence = calculate_confidence(model, X_scaled, "rul")
            soh_estimate = capacity_retention_rate * 100

            # --- Display Results ---
            st.markdown('<div class="section-header">📊 Prediction Results</div>', unsafe_allow_html=True)

            # Gauge
            max_rul = max(1000, rul_pred * 1.5)
            fig_gauge = _create_rul_gauge(rul_pred, max_rul)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Metrics
            m1, m2 = st.columns(2)
            with m1:
                # Estimated lifespan
                total_life = cycles + rul_pred
                st.markdown(
                    f'<div style="text-align:center;padding:12px">'
                    f'<span style="color:{COLORS["muted"]};font-size:0.85rem">ESTIMATED TOTAL LIFESPAN</span><br>'
                    f'<span style="color:{COLORS["accent"]};font-size:1.6rem;font-weight:700">{total_life:.0f} cycles</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f'<div style="text-align:center;padding:12px">'
                    f'<span style="color:{COLORS["muted"]};font-size:0.85rem">PREDICTION CONFIDENCE</span><br>'
                    f'<span style="color:{COLORS["accent"]};font-size:1.6rem;font-weight:700">🎯 {confidence:.1f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        else:
            st.markdown('<div class="section-header">📊 Prediction Results</div>', unsafe_allow_html=True)
            st.info("👈 Enter battery parameters and click **Predict Remaining Life** to see results.")

    # --- Forecast Chart ---
    if submitted:
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📉 Degradation Forecast</div>', unsafe_allow_html=True)

        fig_forecast = _create_degradation_forecast(soh_estimate, rul_pred, cycles)
        st.plotly_chart(fig_forecast, use_container_width=True)

        # --- Recommendations ---
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">💡 Maintenance Recommendations</div>', unsafe_allow_html=True)

        recommendations = get_maintenance_recommendations(
            soh=soh_estimate, rul=rul_pred, temperature=temperature, resistance=resistance
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
