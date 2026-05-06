"""Prediction helpers for the exported taxi fare model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Union

import joblib
import pandas as pd

from src.features.build_features import MODEL_FEATURES, build_location_pair_column
from src.utils.config import get_project_config, resolve_project_path

ModelPath = Optional[Union[str, Path]]


def load_model(model_path: ModelPath = None) -> dict:
    """Load the serialized model artifact."""
    selected_path = resolve_project_path(model_path) if model_path else get_project_config().model_path
    if not selected_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {selected_path}")
    artifact = joblib.load(selected_path)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise ValueError("Invalid model artifact. Expected a dict with a 'pipeline' key.")
    return artifact


def to_model_frame(input_data: Union[pd.DataFrame, dict, Iterable[dict]]) -> pd.DataFrame:
    """Normalize dict/list/DataFrame input into the model feature frame."""
    if isinstance(input_data, pd.DataFrame):
        frame = input_data.copy()
    elif isinstance(input_data, dict):
        frame = pd.DataFrame([input_data])
    else:
        frame = pd.DataFrame(list(input_data))

    frame.columns = [str(column).upper() for column in frame.columns]
    frame = build_location_pair_column(frame)

    missing = [column for column in MODEL_FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")

    return frame[MODEL_FEATURES]


def predict(model_artifact: dict, input_data: Union[pd.DataFrame, dict, Iterable[dict]]) -> list[float]:
    """Return non-negative price predictions as Python floats."""
    frame = to_model_frame(input_data)
    predictions = model_artifact["pipeline"].predict(frame)
    return [float(max(0.0, value)) for value in predictions]


def predict_one(model_artifact: dict, input_data: dict[str, Any]) -> float:
    """Return one prediction for API usage."""
    return predict(model_artifact, input_data)[0]
