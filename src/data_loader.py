"""Load, engineer features, and split the insurance claims dataset."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    AGE_BINS,
    AGE_LABELS,
    DATA_PATH,
    DROP_COLS,
    MIN_REGION_COUNT,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
    VAL_SIZE,
)


def _extract_first_float(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    match = re.search(r"[\d.]+", str(value))
    return float(match.group()) if match else np.nan


def load_raw() -> pd.DataFrame:
    """Load the raw CSV from disk."""
    return pd.read_csv(DATA_PATH)


def _yes_no_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in (TARGET_COL, "policy_id"):
            continue
        vals = df[col].dropna().unique()
        if len(vals) <= 3 and set(str(v) for v in vals).issubset({"Yes", "No"}):
            cols.append(col)
    return cols


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and derive features for modeling."""
    out = df.copy()

    for col in ("max_torque", "max_power"):
        if col in out.columns:
            out[col] = out[col].apply(_extract_first_float)

    out["age_group"] = pd.cut(
        out["customer_age"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        include_lowest=True,
    ).astype(str)

    for col in _yes_no_columns(out):
        out[col] = out[col].map({"Yes": 1, "No": 0})

    if "region_code" in out.columns:
        counts = out["region_code"].value_counts()
        rare = counts[counts < MIN_REGION_COUNT].index
        out["region_code"] = out["region_code"].astype(str)
        out.loc[out["region_code"].isin(rare), "region_code"] = "Other"

    if "model" in out.columns:
        freq = out["model"].value_counts(normalize=True)
        out["model_freq"] = out["model"].map(freq)
        out = out.drop(columns=["model"])

    return out


def default_feature_columns(df: pd.DataFrame) -> list[str]:
    """Feature set: numeric + categoricals, excluding policy_id, target, age_group."""
    exclude = set(DROP_COLS + [TARGET_COL, "age_group"])
    return [c for c in df.columns if c not in exclude]


def split_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    sensitive_col: str,
) -> dict[str, Any]:
    """Stratified train / validation / test split with sensitive attribute aligned."""
    X = df[feature_cols].copy()
    y = df[TARGET_COL].astype(int)
    sensitive = df[sensitive_col].astype(str)

    holdout = TEST_SIZE + VAL_SIZE
    X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(
        X,
        y,
        sensitive,
        test_size=holdout,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    rel_val = VAL_SIZE / holdout
    X_val, X_test, y_val, y_test, s_val, s_test = train_test_split(
        X_temp,
        y_temp,
        s_temp,
        test_size=1 - rel_val,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "s_train": s_train,
        "X_val": X_val,
        "y_val": y_val,
        "s_val": s_val,
        "X_test": X_test,
        "y_test": y_test,
        "s_test": s_test,
    }


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Numeric scaling + one-hot encoding for categoricals."""
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformers = []
    if num_cols:
        transformers.append(("num", numeric_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", categorical_pipe, cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Human-readable feature names after preprocessing."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return [f"f{i}" for i in range(preprocessor.transform(preprocessor.feature_names_in_).shape[1])]


def load_and_prepare(
    sensitive_col: str = "region_code",
    feature_cols: list[str] | None = None,
) -> dict[str, Any]:
    """End-to-end: load, engineer, split."""
    df = engineer_features(load_raw())
    if feature_cols is None:
        feature_cols = default_feature_columns(df)
    splits = split_data(df, feature_cols, sensitive_col)
    preprocessor = build_preprocessor(splits["X_train"])
    return {
        "df": df,
        "feature_cols": feature_cols,
        "sensitive_col": sensitive_col,
        "preprocessor": preprocessor,
        **splits,
    }
