"""Reusable feature engineering for the taxi fare model."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "TOTAL_AMOUNT"

NUMERIC_FEATURES = [
    "TRIP_DISTANCE",
    "PASSENGER_COUNT",
    "PICKUP_HOUR",
    "DAY_OF_WEEK",
    "MONTH",
    "YEAR",
    "IS_WEEKEND",
    "SAME_BOROUGH_FLAG",
    "AIRPORT_TRIP_FLAG",
]

LOW_CARDINALITY_CATEGORICAL_FEATURES = [
    "SERVICE_TYPE",
    "VENDOR_ID",
    "RATE_CODE_ID",
    "TRIP_TYPE",
    "PICKUP_TIME_BAND",
    "PU_BOROUGH",
    "DO_BOROUGH",
]

HIGH_CARDINALITY_BASE_FEATURES = [
    "PU_LOCATION_ID",
    "DO_LOCATION_ID",
    "LOCATION_PAIR",
]

MODEL_FEATURES = (
    NUMERIC_FEATURES
    + LOW_CARDINALITY_CATEGORICAL_FEATURES
    + HIGH_CARDINALITY_BASE_FEATURES
)

ENGINEERED_NUMERIC_FEATURES = NUMERIC_FEATURES
ENGINEERED_CATEGORICAL_FEATURES = LOW_CARDINALITY_CATEGORICAL_FEATURES + [
    "PU_LOCATION_ID",
    "DO_LOCATION_ID",
    "LOCATION_PAIR_TOP",
]
ENGINEERED_FEATURES = ENGINEERED_NUMERIC_FEATURES + ENGINEERED_CATEGORICAL_FEATURES


def _format_location_id(value: object) -> str:
    """Return stable string IDs for Snowflake, pandas and API values."""
    if pd.isna(value):
        return "Unknown"
    try:
        numeric_value = float(value)
        if numeric_value.is_integer():
            return str(int(numeric_value))
    except (TypeError, ValueError):
        pass
    return str(value)


def make_location_pair(pu_location_id: object, do_location_id: object) -> str:
    """Build the route key used by the model."""
    return f"{_format_location_id(pu_location_id)}_{_format_location_id(do_location_id)}"


def ensure_model_columns(df: pd.DataFrame, columns: Sequence[str] = MODEL_FEATURES) -> None:
    """Raise a clear error when required model features are missing."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing model feature columns: {missing}")


def build_location_pair_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add LOCATION_PAIR when only pickup/dropoff IDs are available."""
    out = df.copy()
    if "LOCATION_PAIR" not in out.columns:
        required = {"PU_LOCATION_ID", "DO_LOCATION_ID"}
        missing = sorted(required - set(out.columns))
        if missing:
            raise ValueError(f"Cannot build LOCATION_PAIR. Missing columns: {missing}")
        out["LOCATION_PAIR"] = [
            make_location_pair(pu, do)
            for pu, do in zip(out["PU_LOCATION_ID"], out["DO_LOCATION_ID"])
        ]
    return out


class TripFeatureEngineer(BaseEstimator, TransformerMixin):
    """Feature transformer shared by notebooks, training scripts and the API."""

    def __init__(self, top_location_pairs: Optional[Iterable[str]] = None):
        self.top_location_pairs = set(str(item) for item in (top_location_pairs or []))

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TripFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = build_location_pair_column(pd.DataFrame(X).copy())
        ensure_model_columns(out)

        for column in NUMERIC_FEATURES:
            out[column] = pd.to_numeric(out[column], errors="coerce")

        for column in LOW_CARDINALITY_CATEGORICAL_FEATURES:
            out[column] = out[column].astype("string").fillna("Unknown")

        out["PU_LOCATION_ID"] = out["PU_LOCATION_ID"].map(_format_location_id)
        out["DO_LOCATION_ID"] = out["DO_LOCATION_ID"].map(_format_location_id)
        out["LOCATION_PAIR"] = out["LOCATION_PAIR"].astype("string").fillna("Unknown")
        out["LOCATION_PAIR_TOP"] = np.where(
            out["LOCATION_PAIR"].isin(self.top_location_pairs),
            out["LOCATION_PAIR"],
            "OTHER",
        )

        return out[ENGINEERED_FEATURES]


def preprocess_data(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    """Apply the same basic validity rules used by the Snowflake model table."""
    out = build_location_pair_column(pd.DataFrame(df).copy())

    if target_column in out.columns:
        out[target_column] = pd.to_numeric(out[target_column], errors="coerce")
        out = out[out[target_column].between(0.01, 500, inclusive="both")]

    out["TRIP_DISTANCE"] = pd.to_numeric(out["TRIP_DISTANCE"], errors="coerce")
    out = out[out["TRIP_DISTANCE"].between(0.01, 100, inclusive="both")]

    out["PICKUP_HOUR"] = pd.to_numeric(out["PICKUP_HOUR"], errors="coerce")
    out["DAY_OF_WEEK"] = pd.to_numeric(out["DAY_OF_WEEK"], errors="coerce")
    out = out[out["PICKUP_HOUR"].between(0, 23, inclusive="both")]
    out = out[out["DAY_OF_WEEK"].between(1, 7, inclusive="both")]

    passenger_count = pd.to_numeric(out["PASSENGER_COUNT"], errors="coerce")
    out = out[passenger_count.isna() | passenger_count.between(1, 6, inclusive="both")]

    return out.reset_index(drop=True)


def make_preprocessor() -> ColumnTransformer:
    """Create the sklearn preprocessing block used by all model candidates."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=50,
                    sparse_output=True,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, ENGINEERED_NUMERIC_FEATURES),
            ("cat", categorical_pipeline, ENGINEERED_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def get_feature_pipeline(
    top_location_pairs: Optional[Iterable[str]] = None,
) -> Pipeline:
    """Return the fitted-ready feature pipeline without a final estimator."""
    return Pipeline(
        steps=[
            ("feature_engineering", TripFeatureEngineer(top_location_pairs)),
            ("preprocessor", make_preprocessor()),
        ]
    )


def make_model_pipeline(
    model: BaseEstimator,
    top_location_pairs: Optional[Iterable[str]] = None,
) -> Pipeline:
    """Attach a regressor to the production feature pipeline."""
    return Pipeline(
        steps=[
            ("feature_engineering", TripFeatureEngineer(top_location_pairs)),
            ("preprocessor", make_preprocessor()),
            ("model", model),
        ]
    )
