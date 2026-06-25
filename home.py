"""
Battery Intelligence Platform — Home Dashboard
=================================================
Main landing page with KPIs, fleet health overview,
battery status distribution, and recent analysis summary.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from src.utils import (
    generate_fleet_data, classify_risk, get_risk_color, get_risk_icon,
    RISK_COLORS, RISK_THRESHOLDS,
)
from src.model_evaluation import COLORS, _plotly_dark_layout


def render():
    """Render the Home Dashboard page."""

    # --- Hero Header ---
    st.markdown(
        '<p class="hero-title">⚡ Battery Intelligence Dashboard</p>'
        '<p class="hero-subtitle">Real-time EV battery health monitoring and predictive analytics</p>',
        unsafe_allow_html=True,
    )

    # --- Load fleet data ---
    if "fleet_data" not in st.session_state:
        st.session_state.fleet_data = generate_fleet_data(25)
    fleet = st.session_state.fleet_data

    # --- KPI Cards ---
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.metric(
            label="🔋 Total Batteries",
            value=len(fleet),
            delta="+3 this week",
        )
    with kpi2:
        avg_soh = fleet["SoH (%)"].mean()
        st.metric(
            label="💚 Avg Battery Health",
            value=f"{avg_soh:.1f}%",
            delta=f"{-0.3:.1f}% vs last month",
        )
    with kpi3:
        avg_rul = fleet["RUL (Cycles)"].mean()
        st.metric(
            label="🔄 Avg Remaining Life",
            value=f"{avg_rul:.0f} cycles",
            delta=f"{-12:.0f} vs last month",
        )
    with kpi4:
        critical_count = len(fleet[fleet["Status"] == "Critical"])
        st.metric(
            label="🚨 Critical Alerts",
            value=critical_count,
            delta=f"+{1} new" if critical_count > 0 else "None",
            delta_color="inverse",
        )

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # --- Fleet Health & Status Distribution ---
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown('<div class="section-header">📊 Fleet Health Overview</div>', unsafe_allow_html=True)
        st.markdown('<p class="section-subheader">Battery health distribution across the fleet</p>', unsafe_allow_html=True)

        # Status counts
        status_counts = fleet["Status"].value_counts()
        categories = ["Excellent", "Good", "Warning", "Critical"]
        counts = [status_counts.get(c, 0) for c in categories]
        colors = [RISK_COLORS[c] for c in categories]

        fig_pie = go.Figure(data=[go.Pie(
            labels=categories,
            values=counts,
            hole=0.55,
            marker=dict(colors=colors, line=dict(color=COLORS["bg"], width=2)),
            textinfo="label+value",
            textfont=dict(size=13, family="Inter"),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
        )])
        fig_pie = _plotly_dark_layout(fig_pie)
        fig_pie.update_layout(
            height=350,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            annotations=[dict(
                text=f"<b>{len(fleet)}</b><br>Batteries",
                x=0.5, y=0.5, font_size=16, font_color=COLORS["text"],
                showarrow=False,
            )],
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">📈 Status Distribution</div>', unsafe_allow_html=True)
        st.markdown('<p class="section-subheader">Number of batteries by health category</p>', unsafe_allow_html=True)

        fig_bar = go.Figure()
        for i, cat in enumerate(categories):
            fig_bar.add_trace(go.Bar(
                x=[cat],
                y=[counts[i]],
                marker=dict(color=colors[i], cornerradius=8, line=dict(width=0)),
                text=[str(counts[i])],
                textposition="outside",
                textfont=dict(color=COLORS["text"], size=14, family="Inter"),
                name=cat,
                showlegend=False,
                hovertemplate=f"<b>{cat}</b><br>Count: {counts[i]}<extra></extra>",
            ))

        fig_bar = _plotly_dark_layout(fig_bar)
        fig_bar.update_layout(
            height=350,
            yaxis_title="Count",
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- Health Trend Simulation ---
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📉 Fleet Health Trend (Last 12 Months)</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-subheader">Simulated monthly average SoH across the fleet</p>', unsafe_allow_html=True)

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    # Simulate a gradual decline
    np.random.seed(123)
    base_soh = 94.0
    trend = [base_soh - i * 0.35 + np.random.normal(0, 0.3) for i in range(12)]

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=months,
        y=trend,
        mode="lines+markers",
        line=dict(color=COLORS["accent"], width=3, shape="spline"),
        marker=dict(size=8, color=COLORS["accent"], line=dict(width=2, color=COLORS["bg"])),
        fill="tozeroy",
        fillcolor="rgba(0,209,255,0.07)",
        hovertemplate="<b>%{x}</b><br>Avg SoH: %{y:.1f}%<extra></extra>",
    ))
    fig_trend = _plotly_dark_layout(fig_trend, "")
    fig_trend.update_layout(
        height=300,
        yaxis=dict(title="Avg SoH (%)", range=[85, 100], gridcolor=COLORS["grid"]),
        xaxis=dict(gridcolor=COLORS["grid"]),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # --- Quick Stats ---
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">⚡ Quick Fleet Statistics</div>', unsafe_allow_html=True)

    stat1, stat2, stat3, stat4 = st.columns(4)
    with stat1:
        st.metric("Avg Temperature", f"{fleet['Temperature (°C)'].mean():.1f}°C")
    with stat2:
        st.metric("Avg Resistance", f"{fleet['Resistance (Ω)'].mean():.4f} Ω")
    with stat3:
        st.metric("Avg Capacity", f"{fleet['Capacity (Ah)'].mean():.2f} Ah")
    with stat4:
        st.metric("Total Cycles", f"{fleet['Cycles Used'].sum():,}")

    # --- Recent Alerts ---
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🚨 Recent Alerts</div>', unsafe_allow_html=True)

    critical = fleet[fleet["Status"].isin(["Critical", "Warning"])].sort_values("SoH (%)")
    if len(critical) > 0:
        for _, row in critical.head(5).iterrows():
            icon = get_risk_icon(row["Status"])
            color = get_risk_color(row["Status"])
            st.markdown(
                f'<div class="recommendation-card">'
                f'{icon} <b>{row["Battery ID"]}</b> ({row["Vehicle Model"]}) — '
                f'SoH: <span style="color:{color};font-weight:700">{row["SoH (%)"]:.1f}%</span> | '
                f'RUL: {row["RUL (Cycles)"]} cycles | Status: '
                f'<span style="color:{color}">{row["Status"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.success("✅ No critical alerts. All batteries operating normally.")
