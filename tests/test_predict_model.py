from pathlib import Path

import pytest

from src.models.predict_model import load_model, predict_one
from src.utils.config import get_project_config


def test_exported_model_loads_and_predicts():
    model_path = get_project_config(required=False).model_path
    if not Path(model_path).exists():
        pytest.skip("Model artifact is not available locally.")

    artifact = load_model(model_path)
    payload = {
        "TRIP_DISTANCE": 2.5,
        "PASSENGER_COUNT": 1,
        "PICKUP_HOUR": 12,
        "DAY_OF_WEEK": 3,
        "MONTH": 1,
        "YEAR": 2025,
        "IS_WEEKEND": 0,
        "SAME_BOROUGH_FLAG": 1,
        "AIRPORT_TRIP_FLAG": 0,
        "SERVICE_TYPE": "yellow",
        "VENDOR_ID": "2",
        "RATE_CODE_ID": "1",
        "TRIP_TYPE": "1",
        "PICKUP_TIME_BAND": "midday",
        "PU_BOROUGH": "Manhattan",
        "DO_BOROUGH": "Manhattan",
        "PU_LOCATION_ID": 237,
        "DO_LOCATION_ID": 236,
    }

    prediction = predict_one(artifact, payload)

    assert artifact["model_name"] == "gradient_boosting"
    assert prediction > 0
