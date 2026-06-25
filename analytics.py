"""
Battery Intelligence Platform — Analytics Dashboard
======================================================
Interactive analytics: capacity fade trends, temperature impact,
cycle analysis, health distribution, and correlation heatmap.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import os

from src.model_evaluation import COLORS, _plotly_dark_layout
from src.feature_engineering import engineer_features


def _load_battery_data() -> pd.DataFrame:
    """Load and engineer features for the battery dataset."""
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "battery_data.csv"
    )
    if not os.path.exists(data_path):
        return None
    df = pd.read_csv(data_path)
    df = engineer_features(df)
    return df


BATTERY_COLORS = {
    "B0005": "#00D1FF",
    "B0006": "#34D399",
    "B0007": "#FBBF24",
    "B0018": "#A78BFA",
}


def render():
    """Render the Analytics Dashboard page."""

    st.markdown(
        '<p class="hero-title">📊 Analytics Dashboard</p>'
        '<p class="hero-subtitle">Deep dive into battery performance trends and patterns</p>',
        unsafe_allow_html=True,
    )

    df = _load_battery_data()
    if df is None:
        st.warning("⚠️ No data found. Please generate the dataset first.")
        return

    # Battery selector
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    batteries = df["battery_id"].unique().tolist()
    selected_batteries = st.multiselect(
        "Select Batteries", batteries, default=batteries,
        help="Filter analytics by specific batteries"
    )
    df_filtered = df[df["battery_id"].isin(selected_batteries)]

    if len(df_filtered) == 0:
        st.info("Please select at least one battery.")
        return

    # --- Tab Layout ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📉 Capacity Fade", "🌡️ Temperature Impact", "🔄 Cycle Analysis",
        "📊 Health Distribution", "🔗 Correlation Heatmap",
    ])

    # ─── Tab 1: Capacity Fade Trends ───
    with tab1:
        st.markdown('<div class="section-header">📉 Capacity Fade Over Cycles</div>', unsafe_allow_html=True)
        st.markdown('<p class="section-subheader">Track capacity degradation for each battery over its lifecycle</p>', unsafe_allow_html=True)

        fig = go.Figure()
        for bid in selected_batteries:
            bd = df_filtered[df_filtered["battery_id"] == bid]
            color = BATTERY_COLORS.get(bid, COLORS["accent"])
            fig.add_trace(go.Scatter(
                x=bd["cycle"], y=bd["capacity"],
                mode="lines",
                name=bid,
                line=dict(color=color, width=2.5),
                hovertemplate=f"<b>{bid}</b><br>Cycle: %{{x}}<br>Capacity: %{{y:.4f}} Ah<extra></extra>",
            ))

        fig = _plotly_dark_layout(fig, "")
        fig.update_layout(
            xaxis_title="Cycle Number",
            yaxis_title="Capacity (Ah)",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        # SoH over cycles
        st.markdown('<div class="section-header">🔋 State of Health Over Cycles</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        for bid in selected_batteries:
            bd = df_filtered[df_filtered["battery_id"] == bid]
            color = BATTERY_COLORS.get(bid, COLORS["accent"])
            fig2.add_trace(go.Scatter(
                x=bd["cycle"], y=bd["soh"],
                mode="lines",
                name=bid,
                line=dict(color=color, width=2.5),
                hovertemplate=f"<b>{bid}</b><br>Cycle: %{{x}}<br>SoH: %{{y:.1f}}%<extra></extra>",
            ))
        fig2.add_hline(y=70, line_dash="dash", line_color=COLORS["danger"],
                       annotation_text="End-of-Life (70%)", annotation_font_color=COLORS["danger"])
        fig2 = _plotly_dark_layout(fig2, "")
        fig2.update_layout(
            xaxis_title="Cycle Number", yaxis_title="SoH (%)", height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ─── Tab 2: Temperature Impact ───
    with tab2:
        st.markdown('<div class="section-header">🌡️ Temperature Impact on Battery Health</div>', unsafe_allow_html=True)
        st.markdown('<p class="section-subheader">Analyze how operating temperature affects capacity and degradation</p>', unsafe_allow_html=True)

        fig3 = go.Figure()
        for bid in selected_batteries:
            bd = df_filtered[df_filtered["battery_id"] == bid]
            color = BATTERY_COLORS.get(bid, COLORS["accent"])
            fig3.add_trace(go.Scatter(
                x=bd["temperature_measured"], y=bd["capacity"],
                mode="markers",
                name=bid,
                marker=dict(color=color, size=4, opacity=0.5),
                hovertemplate=f"<b>{bid}</b><br>Temp: %{{x:.1f}}°C<br>Capacity: %{{y:.4f}} Ah<extra></extra>",
            ))
        fig3 = _plotly_dark_layout(fig3, "")
        fig3.update_layout(
            xaxis_title="Temperature (°C)", yaxis_title="Capacity (Ah)", height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Temperature over cycles
        st.markdown('<div class="section-header">🌡️ Temperature Trend Over Cycles</div>', unsafe_allow_html=True)
        fig4 = go.Figure()
        for bid in selected_batteries:
            bd = df_filtered[df_filtered["battery_id"] == bid]
            color = BATTERY_COLORS.get(bid, COLORS["accent"])
            fig4.add_trace(go.Scatter(
                x=bd["cycle"], y=bd["temperature_measured"],
                mode="lines",
                name=bid,
                line=dict(color=color, width=1.5),
                opacity=0.7,
            ))
        fig4 = _plotly_dark_layout(fig4, "")
        fig4.update_layout(
            xaxis_title="Cycle Number", yaxis_title="Temperature (°C)", height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ─── Tab 3: Cycle Analysis ───
    with tab3:
        st.markdown('<div class="section-header">🔄 Cycle Count Analysis</div>', unsafe_allow_html=True)
        st.markdown('<p class="section-subheader">Battery cycle efficiency and resistance growth over time</p>', unsafe_allow_html=True)

        # Internal resistance over cycles
        fig5 = go.Figure()
        for bid in selected_batteries:
            bd = df_filtered[df_filtered["battery_id"] == bid]
            color = BATTERY_COLORS.get(bid, COLORS["accent"])
            fig5.add_trace(go.Scatter(
                x=bd["cycle"], y=bd["internal_resistance"],
                mode="lines",
                name=bid,
                line=dict(color=color, width=2.5),
                hovertemplate=f"<b>{bid}</b><br>Cycle: %{{x}}<br>Resistance: %{{y:.5f}} Ω<extra></extra>",
            ))
        fig5 = _plotly_dark_layout(fig5, "Internal Resistance Growth")
        fig5.update_layout(
            xaxis_title="Cycle Number", yaxis_title="Internal Resistance (Ω)", height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig5, use_container_width=True)

        # Cycle efficiency
        st.markdown('<div class="section-header">⚡ Cycle Efficiency Trend</div>', unsafe_allow_html=True)
        fig6 = go.Figure()
        for bid in selected_batteries:
            bd = df_filtered[df_filtered["battery_id"] == bid]
            color = BATTERY_COLORS.get(bid, COLORS["accent"])
            fig6.add_trace(go.Scatter(
                x=bd["cycle"], y=bd["cycle_efficiency"],
                mode="lines",
                name=bid,
                line=dict(color=color, width=2),
                opacity=0.8,
            ))
        fig6 = _plotly_dark_layout(fig6, "")
        fig6.update_layout(
            xaxis_title="Cycle Number", yaxis_title="Cycle Efficiency", height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig6, use_container_width=True)

    # ─── Tab 4: Health Distribution ───
    with tab4:
        st.markdown('<div class="section-header">📊 Health Distribution</div>', unsafe_allow_html=True)
        st.markdown('<p class="section-subheader">Distribution of SoH values across all cycles</p>', unsafe_allow_html=True)

        fig7 = go.Figure()
        for bid in selected_batteries:
            bd = df_filtered[df_filtered["battery_id"] == bid]
            color = BATTERY_COLORS.get(bid, COLORS["accent"])
            fig7.add_trace(go.Histogram(
                x=bd["soh"],
                name=bid,
                marker_color=color,
                opacity=0.7,
                nbinsx=40,
            ))
        fig7 = _plotly_dark_layout(fig7, "")
        fig7.update_layout(
            xaxis_title="SoH (%)", yaxis_title="Count",
            barmode="overlay", height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig7, use_container_width=True)

        # Box plot
        st.markdown('<div class="section-header">📦 SoH by Battery (Box Plot)</div>', unsafe_allow_html=True)
        fig8 = go.Figure()
        for bid in selected_batteries:
            bd = df_filtered[df_filtered["battery_id"] == bid]
            color = BATTERY_COLORS.get(bid, COLORS["accent"])
            fig8.add_trace(go.Box(
                y=bd["soh"], name=bid,
                marker_color=color,
                line_color=color,
                boxmean="sd",
            ))
        fig8 = _plotly_dark_layout(fig8, "")
        fig8.update_layout(yaxis_title="SoH (%)", height=380, showlegend=False)
        st.plotly_chart(fig8, use_container_width=True)

    # ─── Tab 5: Correlation Heatmap ───
    with tab5:
        st.markdown('<div class="section-header">🔗 Feature Correlation Heatmap</div>', unsafe_allow_html=True)
        st.markdown('<p class="section-subheader">Pearson correlation between key features and target variables</p>', unsafe_allow_html=True)

        corr_cols = [
            "cycle", "voltage_measured", "current_measured", "temperature_measured",
            "capacity", "internal_resistance", "capacity_retention_rate",
            "resistance_growth_rate", "cycle_efficiency", "degradation_rate",
            "temperature_stress_score", "battery_wear_index", "soh", "rul",
        ]
        available_cols = [c for c in corr_cols if c in df_filtered.columns]
        corr_matrix = df_filtered[available_cols].corr()

        # Pretty labels
        labels = [c.replace("_", " ").title() for c in available_cols]

        fig9 = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=labels,
            y=labels,
            colorscale=[
                [0.0, "#F87171"],
                [0.25, "#1E2A3A"],
                [0.5, "#131B2E"],
                [0.75, "#0E3A5E"],
                [1.0, "#00D1FF"],
            ],
            zmin=-1, zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=9, color=COLORS["text"]),
            hovertemplate="%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>",
            colorbar=dict(
                title=dict(text="Corr", font=dict(color=COLORS["text"])),
                tickfont=dict(color=COLORS["muted"]),
            ),
        ))
        fig9 = _plotly_dark_layout(fig9, "")
        fig9.update_layout(
            height=600,
            xaxis=dict(tickangle=45, tickfont=dict(size=10)),
            yaxis=dict(tickfont=dict(size=10)),
        )
        st.plotly_chart(fig9, use_container_width=True)
