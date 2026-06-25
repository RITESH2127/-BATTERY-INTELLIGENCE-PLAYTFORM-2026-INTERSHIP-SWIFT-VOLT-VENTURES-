"""
Battery Intelligence Platform — Fleet Monitoring Dashboard
=============================================================
Simulates a fleet of 25 EV batteries with search, filter,
sort, drill-down, and color-coded status indicators.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

from src.utils import (
    generate_fleet_data, classify_risk, get_risk_color, get_risk_icon,
    get_maintenance_recommendations, RISK_COLORS,
)
from src.model_evaluation import COLORS, _plotly_dark_layout


def style_status(val):
    color = RISK_COLORS.get(val, "#E8EAED")
    return f"color: {color}; font-weight: 700"


def render():
    """Render the Fleet Monitoring Dashboard page."""

    st.markdown(
        '<p class="hero-title">🚗 Fleet Monitoring Dashboard</p>'
        '<p class="hero-subtitle">Real-time monitoring of EV battery fleet health and status</p>',
        unsafe_allow_html=True,
    )

    # --- Load fleet data ---
    if "fleet_data" not in st.session_state:
        st.session_state.fleet_data = generate_fleet_data(25)
    fleet = st.session_state.fleet_data.copy()

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # --- Overview KPIs ---
    status_counts = fleet["Status"].value_counts()
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        excellent = status_counts.get("Excellent", 0)
        st.metric("✅ Excellent", excellent)
    with k2:
        good = status_counts.get("Good", 0)
        st.metric("🔵 Good", good)
    with k3:
        warning = status_counts.get("Warning", 0)
        st.metric("⚠️ Warning", warning)
    with k4:
        critical = status_counts.get("Critical", 0)
        st.metric("🔴 Critical", critical)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # --- Search & Filters ---
    st.markdown('<div class="section-header">🔍 Search & Filter</div>', unsafe_allow_html=True)

    filter_col1, filter_col2, filter_col3 = st.columns([2, 1.5, 1.5])

    with filter_col1:
        search_query = st.text_input(
            "🔎 Search Battery ID",
            placeholder="e.g., EV-0001",
            help="Search by Battery ID",
        )

    with filter_col2:
        status_filter = st.multiselect(
            "📋 Filter by Status",
            ["Excellent", "Good", "Warning", "Critical"],
            default=["Excellent", "Good", "Warning", "Critical"],
        )

    with filter_col3:
        sort_by = st.selectbox(
            "📊 Sort By",
            ["Battery ID", "SoH (%)", "RUL (Cycles)", "Status", "Temperature (°C)"],
            index=1,
        )
        sort_order = st.checkbox("Ascending", value=False)

    # Apply filters
    filtered = fleet[fleet["Status"].isin(status_filter)]
    if search_query:
        filtered = filtered[
            filtered["Battery ID"].str.contains(search_query.upper(), case=False, na=False)
        ]
    filtered = filtered.sort_values(sort_by, ascending=sort_order).reset_index(drop=True)

    st.markdown(f"*Showing {len(filtered)} of {len(fleet)} batteries*")

    # --- Fleet Visualization ---
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    viz_col1, viz_col2 = st.columns([1, 1])

    with viz_col1:
        st.markdown('<div class="section-header">📊 Fleet SoH Distribution</div>', unsafe_allow_html=True)

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=fleet["SoH (%)"],
            nbinsx=15,
            marker=dict(
                color=COLORS["accent"],
                line=dict(width=0.5, color=COLORS["card"]),
            ),
            opacity=0.85,
            hovertemplate="SoH: %{x:.1f}%<br>Count: %{y}<extra></extra>",
        ))
        fig_hist = _plotly_dark_layout(fig_hist, "")
        fig_hist.update_layout(
            xaxis_title="SoH (%)", yaxis_title="Count", height=320,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with viz_col2:
        st.markdown('<div class="section-header">🔋 SoH vs RUL</div>', unsafe_allow_html=True)

        fig_scatter = go.Figure()
        for status in ["Excellent", "Good", "Warning", "Critical"]:
            mask = fleet["Status"] == status
            if mask.any():
                fig_scatter.add_trace(go.Scatter(
                    x=fleet.loc[mask, "SoH (%)"],
                    y=fleet.loc[mask, "RUL (Cycles)"],
                    mode="markers",
                    name=status,
                    marker=dict(
                        color=RISK_COLORS[status],
                        size=10,
                        opacity=0.8,
                        line=dict(width=1, color=COLORS["bg"]),
                    ),
                    text=fleet.loc[mask, "Battery ID"],
                    hovertemplate="<b>%{text}</b><br>SoH: %{x:.1f}%<br>RUL: %{y} cycles<extra></extra>",
                ))
        fig_scatter = _plotly_dark_layout(fig_scatter, "")
        fig_scatter.update_layout(
            xaxis_title="SoH (%)", yaxis_title="RUL (Cycles)", height=320,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # --- Fleet Data Table ---
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📋 Fleet Battery Details</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-subheader">Click on a battery row below for detailed inspection</p>',
        unsafe_allow_html=True,
    )

    styled_df = filtered.style.map(style_status, subset=["Status"])
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        height=400,
        column_config={
            "SoH (%)": st.column_config.ProgressColumn(
                "SoH (%)",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            ),
            "RUL (Cycles)": st.column_config.NumberColumn(
                "RUL (Cycles)",
                format="%d",
            ),
            "Temperature (°C)": st.column_config.NumberColumn(
                "Temperature (°C)",
                format="%.1f°C",
            ),
        },
    )

    # --- Battery Drill-down ---
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🔎 Battery Inspection</div>', unsafe_allow_html=True)

    selected_battery = st.selectbox(
        "Select a battery for detailed inspection",
        filtered["Battery ID"].tolist(),
    )

    if selected_battery:
        bat = filtered[filtered["Battery ID"] == selected_battery].iloc[0]
        status = bat["Status"]
        color = get_risk_color(status)
        icon = get_risk_icon(status)

        # Battery details card
        st.markdown(
            f'<div class="glass-card">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">'
            f'<div>'
            f'<span style="font-size:1.4rem;font-weight:700;color:{COLORS["text"]}">'
            f'🔋 {bat["Battery ID"]}</span><br>'
            f'<span style="color:{COLORS["muted"]}">{bat["Vehicle Model"]}</span>'
            f'</div>'
            f'<span style="color:{color};font-size:1.2rem;font-weight:700">{icon} {status}</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Metrics
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.metric("SoH", f"{bat['SoH (%)']:.1f}%")
        with d2:
            st.metric("RUL", f"{bat['RUL (Cycles)']} cycles")
        with d3:
            st.metric("Temperature", f"{bat['Temperature (°C)']:.1f}°C")
        with d4:
            st.metric("Resistance", f"{bat['Resistance (Ω)']:.4f} Ω")

        d5, d6 = st.columns(2)
        with d5:
            st.metric("Capacity", f"{bat['Capacity (Ah)']:.3f} Ah")
        with d6:
            st.metric("Cycles Used", f"{bat['Cycles Used']}")

        # Recommendations
        st.markdown('<div class="section-header" style="margin-top:16px">💡 Recommendations</div>', unsafe_allow_html=True)
        recs = get_maintenance_recommendations(
            soh=bat["SoH (%)"],
            rul=float(bat["RUL (Cycles)"]),
            temperature=bat["Temperature (°C)"],
            resistance=bat["Resistance (Ω)"],
        )
        for rec in recs:
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
