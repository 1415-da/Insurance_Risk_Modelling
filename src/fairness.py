"""Fairness metrics and mitigation (reweighing, group thresholds)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

try:
    from fairlearn.metrics import (
        demographic_parity_difference,
        demographic_parity_ratio,
        equalized_odds_difference,
    )

    HAS_FAIRLEARN = True
except ImportError:
    HAS_FAIRLEARN = False


FAIRNESS_EXPLANATIONS = {
    "statistical_parity_difference": (
        "Difference between the highest and lowest group selection rates "
        "(share predicted as a claim). Closer to 0 means more parity."
    ),
    "disparate_impact": (
        "Ratio of minimum to maximum selection rate across groups. "
        "1.0 is perfect parity; values below 0.8 are often flagged in hiring audits "
        "(analogy only—not a legal standard for insurance)."
    ),
    "equal_opportunity_difference": (
        "Difference between the highest and lowest true positive rate (TPR) "
        "among people who actually had a claim. Closer to 0 means similar recall across groups."
    ),
    "equalized_odds_difference": (
        "Maximum of TPR and FPR disparities across groups. "
        "Closer to 0 means similar error patterns for all groups."
    ),
}


def _group_mask(sensitive: pd.Series, group: str) -> np.ndarray:
    return sensitive.astype(str).values == str(group)


def selection_rate(y_pred: np.ndarray, sensitive: pd.Series, group: str) -> float:
    mask = _group_mask(sensitive, group)
    if mask.sum() == 0:
        return float("nan")
    return float(y_pred[mask].mean())


def true_positive_rate(
    y_true: np.ndarray, y_pred: np.ndarray, sensitive: pd.Series, group: str
) -> float:
    mask = _group_mask(sensitive, group) & (y_true == 1)
    if mask.sum() == 0:
        return float("nan")
    return float((y_pred[mask] == 1).mean())


def false_positive_rate(
    y_true: np.ndarray, y_pred: np.ndarray, sensitive: pd.Series, group: str
) -> float:
    mask = _group_mask(sensitive, group) & (y_true == 0)
    if mask.sum() == 0:
        return float("nan")
    return float((y_pred[mask] == 1).mean())


def statistical_parity_difference_manual(
    y_pred: np.ndarray, sensitive: pd.Series
) -> float:
    rates = [selection_rate(y_pred, sensitive, g) for g in sensitive.unique()]
    rates = [r for r in rates if not np.isnan(r)]
    return max(rates) - min(rates) if rates else float("nan")


def disparate_impact_manual(y_pred: np.ndarray, sensitive: pd.Series) -> float:
    rates = [selection_rate(y_pred, sensitive, g) for g in sensitive.unique()]
    rates = [r for r in rates if not np.isnan(r) and r > 0]
    if not rates:
        return float("nan")
    return min(rates) / max(rates)


def equal_opportunity_difference_manual(
    y_true: np.ndarray, y_pred: np.ndarray, sensitive: pd.Series
) -> float:
    tprs = [
        true_positive_rate(y_true, y_pred, sensitive, g) for g in sensitive.unique()
    ]
    tprs = [t for t in tprs if not np.isnan(t)]
    return max(tprs) - min(tprs) if tprs else float("nan")


def group_metrics_table(
    y_true: np.ndarray, y_pred: np.ndarray, sensitive: pd.Series
) -> pd.DataFrame:
    """Per-group selection rate, TPR, FPR, and label prevalence."""
    rows = []
    for group in sorted(sensitive.unique(), key=str):
        mask = _group_mask(sensitive, group)
        rows.append(
            {
                "group": str(group),
                "count": int(mask.sum()),
                "selection_rate": selection_rate(y_pred, sensitive, group),
                "tpr": true_positive_rate(y_true, y_pred, sensitive, group),
                "fpr": false_positive_rate(y_true, y_pred, sensitive, group),
                "prevalence": float(y_true[mask].mean()) if mask.sum() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def fairness_summary(
    y_true: np.ndarray, y_pred: np.ndarray, sensitive: pd.Series
) -> dict[str, float]:
    """Aggregate fairness metrics (Fairlearn with manual fallbacks)."""
    y_true_s = pd.Series(y_true)
    y_pred_s = pd.Series(y_pred)
    sensitive_s = sensitive.reset_index(drop=True)

    summary: dict[str, float] = {
        "equal_opportunity_difference": equal_opportunity_difference_manual(
            y_true, y_pred, sensitive
        ),
    }

    if HAS_FAIRLEARN:
        try:
            summary["statistical_parity_difference"] = float(
                demographic_parity_difference(
                    y_true_s, y_pred_s, sensitive_features=sensitive_s
                )
            )
            summary["disparate_impact"] = float(
                demographic_parity_ratio(
                    y_true_s, y_pred_s, sensitive_features=sensitive_s
                )
            )
            summary["equalized_odds_difference"] = float(
                equalized_odds_difference(
                    y_true_s, y_pred_s, sensitive_features=sensitive_s
                )
            )
        except Exception:
            summary["statistical_parity_difference"] = (
                statistical_parity_difference_manual(y_pred, sensitive)
            )
            summary["disparate_impact"] = disparate_impact_manual(y_pred, sensitive)
    else:
        summary["statistical_parity_difference"] = (
            statistical_parity_difference_manual(y_pred, sensitive)
        )
        summary["disparate_impact"] = disparate_impact_manual(y_pred, sensitive)

    return summary


def reweighing_weights(y: np.ndarray, sensitive: np.ndarray) -> np.ndarray:
    """
    AIF360-style reweighing: weight inversely proportional to joint (group, label) frequency.
    """
    y = np.asarray(y)
    s = np.asarray(sensitive)
    n = len(y)
    weights = np.ones(n, dtype=float)
    p_y = y.mean()

    for group in np.unique(s):
        p_a = (s == group).mean()
        for label in (0, 1):
            mask = (s == group) & (y == label)
            p_ay = mask.mean()
            if p_ay > 0:
                p_label = p_y if label == 1 else (1 - p_y)
                weights[mask] = (p_a * p_label) / p_ay

    return weights / weights.mean()


def apply_group_thresholds(
    proba: np.ndarray, sensitive: pd.Series, thresholds: dict[str, float]
) -> np.ndarray:
    """Apply group-specific decision thresholds to probabilities."""
    pred = np.zeros(len(proba), dtype=int)
    for group, thresh in thresholds.items():
        mask = _group_mask(sensitive, group)
        pred[mask] = (proba[mask] >= thresh).astype(int)
    return pred


def tune_group_thresholds(
    y_true: np.ndarray,
    proba: np.ndarray,
    sensitive: pd.Series,
    grid: np.ndarray | None = None,
    max_iter: int = 3,
) -> dict[str, float]:
    """
    Coordinate descent: per-group thresholds to reduce equal opportunity difference.
    """
    if grid is None:
        grid = np.linspace(0.1, 0.9, 17)

    groups = sorted(sensitive.unique(), key=str)
    thresholds = {g: 0.5 for g in groups}

    for _ in range(max_iter):
        for group in groups:
            best_eo = float("inf")
            best_t = thresholds[group]
            for t in grid:
                trial = thresholds.copy()
                trial[group] = float(t)
                pred = apply_group_thresholds(proba, sensitive, trial)
                eo = equal_opportunity_difference_manual(y_true, pred, sensitive)
                if eo < best_eo:
                    best_eo = eo
                    best_t = float(t)
            thresholds[group] = best_t

    return thresholds


def tradeoff_curve(
    y_true: np.ndarray,
    proba: np.ndarray,
    sensitive: pd.Series,
) -> pd.DataFrame:
    """Global threshold sweep: accuracy vs fairness metrics."""
    points = []
    for t in np.linspace(0.05, 0.95, 30):
        pred = (proba >= t).astype(int)
        fair = fairness_summary(y_true, pred, sensitive)
        points.append(
            {
                "threshold": t,
                "accuracy": accuracy_score(y_true, pred),
                **fair,
            }
        )
    return pd.DataFrame(points)


def mitigation_comparison(
    y_true: np.ndarray,
    proba: np.ndarray,
    sensitive: pd.Series,
    baseline_pred: np.ndarray,
    mitigated_pred: np.ndarray,
) -> pd.DataFrame:
    """Before/after metrics for mitigation UI."""
    rows = []
    for label, pred in [("Baseline (0.5 threshold)", baseline_pred), ("After mitigation", mitigated_pred)]:
        metrics = fairness_summary(y_true, pred, sensitive)
        rows.append(
            {
                "model": label,
                "accuracy": accuracy_score(y_true, pred),
                **metrics,
            }
        )
    return pd.DataFrame(rows)
