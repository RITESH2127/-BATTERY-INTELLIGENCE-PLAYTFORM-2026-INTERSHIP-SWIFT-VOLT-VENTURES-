"""
Battery Intelligence Platform — Explainable AI
=================================================
SHAP-based model explanations for SoH and RUL predictions.
Provides global feature importance and local prediction breakdowns.
"""

import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go

from src.model_evaluation import COLORS, _plotly_dark_layout


def compute_shap_values(model, X: pd.DataFrame) -> tuple:
    """
    Compute SHAP values for a tree-based model.

    Parameters
    ----------
    model : fitted tree-based model (RF, XGBoost, GBR, LGBM)
    X : pd.DataFrame of input features

    Returns
    -------
    tuple
        (shap_values_array, expected_value, feature_names)
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    expected_value = explainer.expected_value
    if isinstance(expected_value, np.ndarray):
        expected_value = expected_value[0]
    return shap_values, expected_value, list(X.columns)


def get_global_feature_importance(shap_values: np.ndarray,
                                   feature_names: list) -> pd.DataFrame:
    """
    Compute global feature importance from SHAP values.

    Returns
    -------
    pd.DataFrame with columns: feature, importance (sorted descending)
    """
    importance = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return df


def create_shap_importance_chart(importance_df: pd.DataFrame,
                                  title: str = "SHAP Feature Importance",
                                  top_n: int = 15) -> go.Figure:
    """Create a horizontal bar chart of SHAP feature importance."""
    df = importance_df.head(top_n).sort_values("importance", ascending=True)

    # Create gradient colors from muted to bright accent
    n = len(df)
    colors = [
        f"rgba(0, {int(140 + 115 * i / max(n-1, 1))}, {int(180 + 75 * i / max(n-1, 1))}, 0.85)"
        for i in range(n)
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["feature"],
        x=df["importance"],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0), cornerradius=4),
        text=[f"{v:.4f}" for v in df["importance"]],
        textposition="outside",
        textfont=dict(color=COLORS["text"], size=11),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))

    fig = _plotly_dark_layout(fig, title)
    fig.update_layout(
        xaxis_title="Mean |SHAP Value|",
        yaxis_title="",
        height=max(400, top_n * 32),
        margin=dict(l=180),
    )
    return fig


def get_local_explanation(shap_values: np.ndarray, expected_value: float,
                          feature_names: list, instance_idx: int) -> pd.DataFrame:
    """
    Get SHAP explanation for a single prediction.

    Returns
    -------
    pd.DataFrame with columns: feature, shap_value, abs_shap_value (sorted by |shap|)
    """
    sv = shap_values[instance_idx]
    df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": sv,
        "abs_shap_value": np.abs(sv),
    }).sort_values("abs_shap_value", ascending=False).reset_index(drop=True)
    return df


def create_waterfall_chart(local_df: pd.DataFrame, expected_value: float,
                           prediction: float, title: str = "Prediction Breakdown",
                           top_n: int = 10) -> go.Figure:
    """Create a waterfall-style chart explaining a single prediction."""
    df = local_df.head(top_n).sort_values("abs_shap_value", ascending=True)

    colors = [COLORS["success"] if v > 0 else COLORS["danger"] for v in df["shap_value"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["feature"],
        x=df["shap_value"],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0), cornerradius=4),
        text=[f"{v:+.3f}" for v in df["shap_value"]],
        textposition="outside",
        textfont=dict(color=COLORS["text"], size=11),
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:+.4f}<extra></extra>",
    ))

    fig = _plotly_dark_layout(fig, title)
    fig.update_layout(
        xaxis_title="SHAP Value (Impact on Prediction)",
        height=max(380, top_n * 35),
        margin=dict(l=180),
        annotations=[
            dict(
                text=f"Base: {expected_value:.1f} → Prediction: {prediction:.1f}",
                xref="paper", yref="paper",
                x=0.98, y=1.08,
                showarrow=False,
                font=dict(size=12, color=COLORS["accent"]),
            )
        ],
    )
    return fig


def create_feature_contribution_table(local_df: pd.DataFrame,
                                       feature_values: pd.Series,
                                       top_n: int = 10) -> pd.DataFrame:
    """
    Create a human-readable feature contribution table.

    Returns
    -------
    pd.DataFrame with columns: Feature, Value, SHAP Impact, Direction
    """
    df = local_df.head(top_n).copy()
    rows = []
    for _, row in df.iterrows():
        feat = row["feature"]
        val = feature_values.get(feat, "N/A")
        if isinstance(val, float):
            val = round(val, 4)
        direction = "⬆ Increases" if row["shap_value"] > 0 else "⬇ Decreases"
        rows.append({
            "Feature": feat.replace("_", " ").title(),
            "Value": val,
            "SHAP Impact": f"{row['shap_value']:+.4f}",
            "Direction": direction,
        })
    return pd.DataFrame(rows)
