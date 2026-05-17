"""Random Forest training, evaluation, and persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from .config import MODELS_DIR, RANDOM_STATE
from .data_loader import build_preprocessor, get_feature_names


def build_rf_pipeline(
    preprocessor,
    n_estimators: int = 200,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    class_weight: str | dict | None = "balanced",
) -> Pipeline:
    """Build sklearn Pipeline: preprocess + RandomForestClassifier."""
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("prep", preprocessor), ("clf", clf)])


def train_model(
    pipeline: Pipeline,
    X_train,
    y_train,
    sample_weight: np.ndarray | None = None,
) -> Pipeline:
    """Fit pipeline; optional sample weights for reweighing mitigation."""
    fit_kwargs: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["clf__sample_weight"] = sample_weight
    pipeline.fit(X_train, y_train, **fit_kwargs)
    return pipeline


def predict_proba(pipeline: Pipeline, X) -> np.ndarray:
    """Positive-class probabilities."""
    return pipeline.predict_proba(X)[:, 1]


def evaluate(
    pipeline: Pipeline,
    X,
    y,
    threshold: float = 0.5,
    y_pred: np.ndarray | None = None,
) -> dict[str, Any]:
    """Classification metrics and curves data."""
    proba = predict_proba(pipeline, X)
    if y_pred is None:
        y_pred = (proba >= threshold).astype(int)

    fpr, tpr, thresholds = roc_curve(y, proba)
    auc = roc_auc_score(y, proba) if len(np.unique(y)) > 1 else 0.0

    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": auc,
        "proba": proba,
        "pred": y_pred,
        "confusion_matrix": confusion_matrix(y, y_pred),
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": thresholds,
    }


def get_feature_importances(pipeline: Pipeline) -> tuple[list[str], np.ndarray]:
    """Feature importances aligned with fitted preprocessor column names."""
    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]
    names = get_feature_names(prep)
    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1]
    return [names[i] for i in order], importances[order]


def save_model(pipeline: Pipeline, name: str = "rf_baseline.joblib") -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / name
    joblib.dump(pipeline, path)
    return path


def load_model(name: str = "rf_baseline.joblib") -> Pipeline:
    return joblib.load(MODELS_DIR / name)


def build_and_train(
    X_train,
    y_train,
    preprocessor=None,
    sample_weight: np.ndarray | None = None,
    **rf_kwargs,
) -> Pipeline:
    """Convenience: create preprocessor if needed, train, return pipeline."""
    if preprocessor is None:
        preprocessor = build_preprocessor(X_train)
    pipe = build_rf_pipeline(preprocessor, **rf_kwargs)
    return train_model(pipe, X_train, y_train, sample_weight=sample_weight)
