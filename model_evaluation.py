"""
Battery Intelligence Platform — Model Evaluation
===================================================
Computes evaluation metrics (MAE, MSE, RMSE, R²) for all trained models
and generates comparison visualizations using Plotly.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Color palette matching the dark theme
COLORS = {
    "accent": "#00D1FF",
    "accent2": "#0088CC",
    "success": "#34D399",
    "warning": "#FBBF24",
    "danger": "#F87171",
    "info": "#60A5FA",
    "bg": "#0B1120",
    "card": "#131B2E",
    "text": "#E8EAED",
    "muted": "#5F6368",
    "grid": "rgba(255,255,255,0.05)",
}

MODEL_COLORS = ["#00D1FF", "#34D399", "#FBBF24", "#F87171", "#60A5FA", "#A78BFA"]


def _plotly_dark_layout(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply the standard dark theme layout to a Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=COLORS["text"]), x=0.0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=COLORS["text"], size=12),
        xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
        legend=dict(
            bgcolor="rgba(19,27,46,0.8)",
            bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=40, r=20, t=50, b=40),
        hoverlabel=dict(bgcolor=COLORS["card"], font_size=12, font_color=COLORS["text"]),
    )
    return fig


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluate a single model and return metrics.

    Returns
    -------
    dict with keys: model_name, mae, mse, rmse, r2, predictions
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return {
        "model_name": model_name,
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "predictions": y_pred,
        "actuals": np.array(y_test),
    }


def evaluate_all_models(models_dict: dict, X_test, y_test) -> list:
    """Evaluate all models and return list of result dicts."""
    results = []
    for name, info in models_dict.items():
        result = evaluate_model(info["model"], X_test, y_test, name)
        results.append(result)
    return results


def create_comparison_chart(results: list, metric: str = "r2", title: str = "Model Comparison") -> go.Figure:
    """Create a bar chart comparing a specific metric across models."""
    names = [r["model_name"] for r in results]
    values = [r[metric] for r in results]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names,
        y=values,
        marker=dict(
            color=MODEL_COLORS[:len(names)],
            line=dict(width=0),
            cornerradius=6,
        ),
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
        textfont=dict(color=COLORS["text"], size=13, family="Inter"),
        hovertemplate="<b>%{x}</b><br>" + f"{metric.upper()}: " + "%{y:.4f}<extra></extra>",
    ))

    fig = _plotly_dark_layout(fig, title)
    fig.update_layout(
        yaxis_title=metric.upper(),
        xaxis_title="Model",
        showlegend=False,
        height=400,
    )
    return fig


def create_all_metrics_chart(results: list, title: str = "Model Performance Comparison") -> go.Figure:
    """Create a grouped bar chart comparing all metrics across models."""
    names = [r["model_name"] for r in results]
    metrics = ["mae", "rmse", "r2"]
    metric_labels = ["MAE", "RMSE", "R²"]
    metric_colors = [COLORS["warning"], COLORS["danger"], COLORS["accent"]]

    fig = go.Figure()
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [r[metric] for r in results]
        fig.add_trace(go.Bar(
            name=label,
            x=names,
            y=values,
            marker_color=metric_colors[i],
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
            textfont=dict(size=10),
        ))

    fig = _plotly_dark_layout(fig, title)
    fig.update_layout(
        barmode="group",
        yaxis_title="Score",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_actual_vs_predicted(result: dict, title: str = "Actual vs Predicted") -> go.Figure:
    """Create a scatter plot of actual vs predicted values."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=result["actuals"],
        y=result["predictions"],
        mode="markers",
        marker=dict(
            color=COLORS["accent"],
            size=5,
            opacity=0.5,
            line=dict(width=0),
        ),
        name="Predictions",
        hovertemplate="Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>",
    ))

    # Perfect prediction line
    min_val = min(result["actuals"].min(), result["predictions"].min())
    max_val = max(result["actuals"].max(), result["predictions"].max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line=dict(color=COLORS["danger"], dash="dash", width=2),
        name="Perfect Prediction",
    ))

    fig = _plotly_dark_layout(fig, f"{title} — {result['model_name']}")
    fig.update_layout(
        xaxis_title="Actual",
        yaxis_title="Predicted",
        height=450,
    )
    return fig


def create_residual_plot(result: dict, title: str = "Residual Distribution") -> go.Figure:
    """Create a histogram of prediction residuals."""
    residuals = result["actuals"] - result["predictions"]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=50,
        marker=dict(color=COLORS["accent"], line=dict(width=0.5, color=COLORS["card"])),
        opacity=0.85,
        hovertemplate="Residual: %{x:.2f}<br>Count: %{y}<extra></extra>",
    ))

    fig = _plotly_dark_layout(fig, f"{title} — {result['model_name']}")
    fig.update_layout(
        xaxis_title="Residual (Actual − Predicted)",
        yaxis_title="Count",
        height=380,
    )
    return fig


def create_metrics_table(results: list) -> pd.DataFrame:
    """Create a summary DataFrame of all model metrics."""
    rows = []
    for r in results:
        rows.append({
            "Model": r["model_name"],
            "MAE": r["mae"],
            "MSE": r["mse"],
            "RMSE": r["rmse"],
            "R² Score": r["r2"],
        })
    return pd.DataFrame(rows).sort_values("R² Score", ascending=False).reset_index(drop=True)
