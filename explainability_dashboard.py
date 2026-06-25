"""
Battery Intelligence Platform — Explainable AI Dashboard
==========================================================
SHAP feature importance, prediction breakdown, and
top factors affecting battery health.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import joblib
import os
import json

from src.explainability import (
    compute_shap_values, get_global_feature_importance,
    create_shap_importance_chart, get_local_explanation,
    create_waterfall_chart, create_feature_contribution_table,
)
from src.model_evaluation import COLORS, _plotly_dark_layout
from src.feature_engineering import SOH_FEATURES, RUL_FEATURES, engineer_features


def _load_models_and_data():
    """Load models, scalers, and test data for SHAP analysis."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    data_path = os.path.join(base_dir, "data", "battery_data.csv")

    if not os.path.exists(os.path.join(models_dir, "model_metadata.json")):
        return None

    with open(os.path.join(models_dir, "model_metadata.json"), "r") as f:
        metadata = json.load(f)

    # Load best SoH model
    soh_name = metadata["soh_best_model"].lower().replace(" ", "_")
    soh_model = joblib.load(os.path.join(models_dir, f"soh_{soh_name}.joblib"))
    soh_scaler = joblib.load(os.path.join(models_dir, "soh_scaler.joblib"))

    # Load best RUL model
    rul_name = metadata["rul_best_model"].lower().replace(" ", "_")
    rul_model = joblib.load(os.path.join(models_dir, f"rul_{rul_name}.joblib"))
    rul_scaler = joblib.load(os.path.join(models_dir, "rul_scaler.joblib"))

    # Load and prepare data
    df = pd.read_csv(data_path)
    df = engineer_features(df)

    return {
        "soh_model": soh_model,
        "rul_model": rul_model,
        "soh_scaler": soh_scaler,
        "rul_scaler": rul_scaler,
        "data": df,
        "metadata": metadata,
    }


def render():
    """Render the Explainable AI Dashboard page."""

    st.markdown(
        '<p class="hero-title">🧠 Explainable AI Dashboard</p>'
        '<p class="hero-subtitle">Understand why the model makes specific predictions using SHAP analysis</p>',
        unsafe_allow_html=True,
    )

    artifacts = _load_models_and_data()
    if artifacts is None:
        st.warning("⚠️ Models not trained yet. Please run the training pipeline first.")
        return

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Task selector
    task = st.radio(
        "Select Prediction Task",
        ["State of Health (SoH)", "Remaining Useful Life (RUL)"],
        horizontal=True,
    )

    is_soh = task.startswith("State")
    model = artifacts["soh_model"] if is_soh else artifacts["rul_model"]
    scaler = artifacts["soh_scaler"] if is_soh else artifacts["rul_scaler"]
    features = SOH_FEATURES if is_soh else RUL_FEATURES
    target = "soh" if is_soh else "rul"
    df = artifacts["data"]

    X = df[features]
    X_scaled = pd.DataFrame(scaler.transform(X), columns=features)

    # Use a sample for SHAP computation (speed)
    sample_size = min(200, len(X_scaled))
    X_sample = X_scaled.sample(n=sample_size, random_state=42)
    X_raw_sample = X.loc[X_sample.index]

    # --- Compute SHAP Values ---
    with st.spinner("Computing SHAP values... This may take a moment."):
        shap_values, expected_value, feature_names = compute_shap_values(model, X_sample)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # --- Tab Layout ---
    tab1, tab2, tab3 = st.tabs([
        "📊 Global Feature Importance",
        "🔍 Individual Prediction Breakdown",
        "📋 Top Factors Analysis",
    ])

    # ─── Tab 1: Global Feature Importance ───
    with tab1:
        st.markdown(
            f'<div class="section-header">📊 SHAP Feature Importance — {task}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="section-subheader">'
            'Mean absolute SHAP value indicates each feature\'s average impact on predictions'
            '</p>',
            unsafe_allow_html=True,
        )

        importance_df = get_global_feature_importance(shap_values, feature_names)
        fig = create_shap_importance_chart(
            importance_df,
            title=f"SHAP Feature Importance — {task}",
            top_n=15,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary table
        st.markdown('<div class="section-header">📋 Feature Importance Table</div>', unsafe_allow_html=True)
        display_df = importance_df.copy()
        display_df["feature"] = display_df["feature"].str.replace("_", " ").str.title()
        display_df["importance"] = display_df["importance"].round(4)
        display_df.columns = ["Feature", "Mean |SHAP Value|"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ─── Tab 2: Individual Prediction Breakdown ───
    with tab2:
        st.markdown(
            '<div class="section-header">🔍 Individual Prediction Explanation</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="section-subheader">'
            'Select a data point to see which features pushed the prediction up or down'
            '</p>',
            unsafe_allow_html=True,
        )

        # Instance selector
        instance_idx = st.slider(
            "Select data point index",
            0, len(X_sample) - 1, 0,
            help="Choose a specific data point to explain",
        )

        prediction = model.predict(X_sample.iloc[[instance_idx]])[0]
        local_df = get_local_explanation(shap_values, expected_value, feature_names, instance_idx)

        # Waterfall chart
        fig_wf = create_waterfall_chart(
            local_df, expected_value, prediction,
            title=f"Prediction Breakdown (Point #{instance_idx})",
            top_n=12,
        )
        st.plotly_chart(fig_wf, use_container_width=True)

        # Feature contribution table
        st.markdown('<div class="section-header">📋 Feature Contributions</div>', unsafe_allow_html=True)
        contrib_df = create_feature_contribution_table(
            local_df,
            X_raw_sample.iloc[instance_idx],
            top_n=12,
        )
        st.dataframe(contrib_df, use_container_width=True, hide_index=True)

    # ─── Tab 3: Top Factors Analysis ───
    with tab3:
        st.markdown(
            f'<div class="section-header">🏆 Top Factors Affecting {task}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="section-subheader">'
            'Key features and their directional impact on predictions'
            '</p>',
            unsafe_allow_html=True,
        )

        importance_df = get_global_feature_importance(shap_values, feature_names)
        top5 = importance_df.head(5)

        for i, (_, row) in enumerate(top5.iterrows()):
            feat_name = row["feature"].replace("_", " ").title()
            imp = row["importance"]

            # Determine average direction
            feat_idx = feature_names.index(row["feature"])
            avg_shap = shap_values[:, feat_idx].mean()
            direction = "⬆️ Increases" if avg_shap > 0 else "⬇️ Decreases"
            dir_color = COLORS["success"] if avg_shap > 0 else COLORS["danger"]

            rank_colors = [COLORS["accent"], "#34D399", "#FBBF24", "#A78BFA", "#60A5FA"]
            accent = rank_colors[i % len(rank_colors)]

            st.markdown(
                f'<div class="recommendation-card" style="border-left-color:{accent}">'
                f'<div style="display:flex;justify-content:space-between;align-items:center">'
                f'<div>'
                f'<span style="color:{accent};font-weight:700;font-size:1.1rem">#{i+1} {feat_name}</span><br>'
                f'<span style="color:{COLORS["muted"]};font-size:0.85rem">'
                f'Mean |SHAP|: {imp:.4f}</span>'
                f'</div>'
                f'<div style="text-align:right">'
                f'<span style="color:{dir_color};font-weight:600">{direction}</span><br>'
                f'<span style="color:{COLORS["muted"]};font-size:0.85rem">prediction on average</span>'
                f'</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Feature interaction: top 2 features scatter
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🔗 Feature Interaction</div>', unsafe_allow_html=True)

        top2 = importance_df.head(2)["feature"].tolist()
        if len(top2) == 2:
            f1_idx = feature_names.index(top2[0])
            f2_idx = feature_names.index(top2[1])

            fig_int = go.Figure()
            fig_int.add_trace(go.Scatter(
                x=X_raw_sample[top2[0]].values,
                y=shap_values[:, f1_idx],
                mode="markers",
                marker=dict(
                    color=X_raw_sample[top2[1]].values,
                    colorscale=[[0, "#3B82F6"], [0.5, "#60A5FA"], [1, "#00D1FF"]],
                    size=6,
                    opacity=0.7,
                    colorbar=dict(
                        title=dict(text=top2[1].replace("_", " ").title(), font=dict(color=COLORS["text"], size=11)),
                        tickfont=dict(color=COLORS["muted"]),
                    ),
                ),
                hovertemplate=(
                    f"{top2[0].replace('_', ' ').title()}: %{{x:.3f}}<br>"
                    f"SHAP Value: %{{y:.4f}}<br>"
                    f"<extra></extra>"
                ),
            ))
            fig_int = _plotly_dark_layout(fig_int, f"SHAP Dependence: {top2[0].replace('_', ' ').title()}")
            fig_int.update_layout(
                xaxis_title=top2[0].replace("_", " ").title(),
                yaxis_title="SHAP Value",
                height=420,
                showlegend=False,
            )
            st.plotly_chart(fig_int, use_container_width=True)
