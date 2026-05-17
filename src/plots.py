"""Visualization helpers for Streamlit (Plotly + Matplotlib)."""

from __future__ import annotations

import io
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.calibration import calibration_curve


def plot_roc(fpr, tpr, auc: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})")
    )
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Chance")
    )
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=420,
    )
    return fig


def plot_confusion_matrix(cm: np.ndarray) -> go.Figure:
    labels = ["No claim", "Claim"]
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            text=cm,
            texttemplate="%{text}",
            colorscale="Blues",
        )
    )
    fig.update_layout(title="Confusion Matrix", height=400)
    return fig


def plot_calibration(y_true, proba) -> go.Figure:
    prob_true, prob_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode="lines+markers", name="Model"))
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Perfect")
    )
    fig.update_layout(
        title="Calibration Plot",
        xaxis_title="Mean predicted probability",
        yaxis_title="Fraction of positives",
        height=420,
    )
    return fig


def plot_probability_distribution(proba, y_true) -> go.Figure:
    df = pd.DataFrame({"probability": proba, "actual": y_true.map({0: "No claim", 1: "Claim"})})
    fig = px.histogram(
        df,
        x="probability",
        color="actual",
        barmode="overlay",
        opacity=0.65,
        nbins=40,
        title="Predicted Probability Distribution",
    )
    fig.update_layout(height=420)
    return fig


def plot_feature_importance(names: list[str], importances: np.ndarray, top_n: int = 20) -> go.Figure:
    n = min(top_n, len(names))
    df = pd.DataFrame({"feature": names[:n], "importance": importances[:n]})
    fig = px.bar(
        df,
        x="importance",
        y="feature",
        orientation="h",
        title=f"Top {n} Feature Importances",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
    return fig


def plot_group_bars(df: pd.DataFrame, metric: str, title: str) -> go.Figure:
    fig = px.bar(
        df,
        x="group",
        y=metric,
        title=title,
        labels={metric: metric.replace("_", " ").title(), "group": "Group"},
    )
    fig.update_layout(xaxis_tickangle=-45, height=420)
    return fig


def plot_claim_rate_by_group(df: pd.DataFrame, group_col: str, target: str) -> go.Figure:
    rates = df.groupby(group_col, observed=True)[target].mean().reset_index()
    rates.columns = ["group", "claim_rate"]
    fig = px.bar(
        rates.sort_values("claim_rate", ascending=False),
        x="group",
        y="claim_rate",
        title=f"Claim Rate by {group_col}",
    )
    fig.update_layout(xaxis_tickangle=-45, height=420)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    corr = df.select_dtypes(include=[np.number]).corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu",
            zmid=0,
        )
    )
    fig.update_layout(title="Correlation Heatmap (numeric features)", height=600)
    return fig


def plot_tradeoff(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    fig = px.scatter(
        df,
        x=x,
        y=y,
        hover_data=["threshold"] if "threshold" in df.columns else None,
        title=title,
    )
    fig.update_layout(height=450)
    return fig


def plot_distribution(df: pd.DataFrame, column: str) -> go.Figure:
    if df[column].dtype in [np.number, "float64", "int64"]:
        fig = px.histogram(df, x=column, nbins=40, title=f"Distribution: {column}")
    else:
        counts = df[column].value_counts().head(20).reset_index()
        counts.columns = [column, "count"]
        fig = px.bar(counts, x=column, y="count", title=f"Counts: {column}")
        fig.update_layout(xaxis_tickangle=-45)
    fig.update_layout(height=400)
    return fig


def shap_summary_figure(pipeline, X_sample: pd.DataFrame, max_display: int = 15):
    """Return matplotlib figure for SHAP summary (or None on failure)."""
    try:
        import shap

        prep = pipeline.named_steps["prep"]
        clf = pipeline.named_steps["clf"]
        X_t = prep.transform(X_sample)
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_t)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_t, show=False, max_display=max_display)
        plt.tight_layout()
        return fig
    except Exception:
        plt.close("all")
        return None


def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()
