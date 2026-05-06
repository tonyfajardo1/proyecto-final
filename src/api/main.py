"""FastAPI service for taxi fare prediction."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date, time
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.models.predict_model import load_model, predict_one

MODEL_ARTIFACT: Optional[dict] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="API - Prediccion de Precios ML",
    version="1.0",
    lifespan=lifespan,
)


class TripInput(BaseModel):
    """Input fields known before or at pickup time."""

    model_config = ConfigDict(populate_by_name=True)

    trip_distance: float = Field(..., alias="TRIP_DISTANCE", gt=0, le=100)
    passenger_count: int = Field(..., alias="PASSENGER_COUNT", ge=1, le=6)
    pickup_hour: int = Field(..., alias="PICKUP_HOUR", ge=0, le=23)
    day_of_week: int = Field(..., alias="DAY_OF_WEEK", ge=1, le=7)
    month: int = Field(..., alias="MONTH", ge=1, le=12)
    year: int = Field(..., alias="YEAR", ge=2015, le=2026)
    is_weekend: int = Field(..., alias="IS_WEEKEND", ge=0, le=1)
    same_borough_flag: int = Field(..., alias="SAME_BOROUGH_FLAG", ge=0, le=1)
    airport_trip_flag: int = Field(..., alias="AIRPORT_TRIP_FLAG", ge=0, le=1)
    service_type: str = Field(..., alias="SERVICE_TYPE")
    vendor_id: str = Field(..., alias="VENDOR_ID")
    rate_code_id: str = Field(..., alias="RATE_CODE_ID")
    trip_type: str = Field("Unknown", alias="TRIP_TYPE")
    pickup_time_band: str = Field(..., alias="PICKUP_TIME_BAND")
    pu_borough: str = Field(..., alias="PU_BOROUGH")
    do_borough: str = Field(..., alias="DO_BOROUGH")
    pu_location_id: int = Field(..., alias="PU_LOCATION_ID", ge=1)
    do_location_id: int = Field(..., alias="DO_LOCATION_ID", ge=1)
    location_pair: Optional[str] = Field(None, alias="LOCATION_PAIR")


class TripSimpleInput(BaseModel):
    """User-facing input. The API derives the technical model fields."""

    pickup_date: date
    pickup_time: time
    trip_distance: float = Field(..., gt=0, le=100)
    passenger_count: int = Field(..., ge=1, le=6)
    service_type: str = "yellow"
    vendor_id: str = "2"
    rate_code_id: str = "1"
    trip_type: str = "1"
    pu_borough: str = "Manhattan"
    do_borough: str = "Manhattan"
    pu_location_id: int = Field(..., ge=1)
    do_location_id: int = Field(..., ge=1)


class PredictionResponse(BaseModel):
    estimated_total_amount: float
    model_name: str


def pickup_time_band(hour: int) -> str:
    """Map pickup hour to the same time band used in Snowflake."""
    if 6 <= hour <= 9:
        return "morning_peak"
    if 10 <= hour <= 15:
        return "midday"
    if 16 <= hour <= 20:
        return "evening_peak"
    if 21 <= hour <= 23:
        return "night"
    return "overnight"


def simple_to_model_payload(trip: TripSimpleInput) -> dict:
    """Convert user-facing input into the model feature contract."""
    pickup_hour = trip.pickup_time.hour
    day_of_week = trip.pickup_date.isoweekday()
    airport_locations = {1, 132, 138}

    return {
        "TRIP_DISTANCE": trip.trip_distance,
        "PASSENGER_COUNT": trip.passenger_count,
        "PICKUP_HOUR": pickup_hour,
        "DAY_OF_WEEK": day_of_week,
        "MONTH": trip.pickup_date.month,
        "YEAR": trip.pickup_date.year,
        "IS_WEEKEND": int(day_of_week in (6, 7)),
        "SAME_BOROUGH_FLAG": int(trip.pu_borough == trip.do_borough),
        "AIRPORT_TRIP_FLAG": int(
            trip.pu_location_id in airport_locations
            or trip.do_location_id in airport_locations
        ),
        "SERVICE_TYPE": trip.service_type,
        "VENDOR_ID": str(trip.vendor_id),
        "RATE_CODE_ID": str(trip.rate_code_id),
        "TRIP_TYPE": str(trip.trip_type),
        "PICKUP_TIME_BAND": pickup_time_band(pickup_hour),
        "PU_BOROUGH": trip.pu_borough,
        "DO_BOROUGH": trip.do_borough,
        "PU_LOCATION_ID": trip.pu_location_id,
        "DO_LOCATION_ID": trip.do_location_id,
    }


def load_artifacts() -> None:
    """Load the model once when the API starts."""
    global MODEL_ARTIFACT
    MODEL_ARTIFACT = load_model()


@app.get("/health")
def health() -> dict:
    """Simple service health check."""
    if MODEL_ARTIFACT is None:
        return {"status": "starting", "model_loaded": False}
    return {
        "status": "ok",
        "model_loaded": True,
        "model_name": MODEL_ARTIFACT.get("model_name", "unknown"),
    }


@app.get("/")
def root() -> dict:
    """Friendly landing response for demo navigation."""
    return {
        "service": "API - Prediccion de Precios ML",
        "health": "/health",
        "docs": "/docs",
        "demo_endpoint": "/predict-simple",
        "technical_endpoint": "/predict",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_price(trip: TripInput) -> PredictionResponse:
    """Predict total trip amount from pickup-time features."""
    global MODEL_ARTIFACT
    if MODEL_ARTIFACT is None:
        MODEL_ARTIFACT = load_model()

    try:
        payload = trip.model_dump(by_alias=True)
        prediction = predict_one(MODEL_ARTIFACT, payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictionResponse(
        estimated_total_amount=round(prediction, 2),
        model_name=str(MODEL_ARTIFACT.get("model_name", "unknown")),
    )


@app.post("/predict-simple", response_model=PredictionResponse)
def predict_price_simple(trip: TripSimpleInput) -> PredictionResponse:
    """Predict from user-facing trip fields for demos and the Streamlit app."""
    global MODEL_ARTIFACT
    if MODEL_ARTIFACT is None:
        MODEL_ARTIFACT = load_model()

    try:
        prediction = predict_one(MODEL_ARTIFACT, simple_to_model_payload(trip))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictionResponse(
        estimated_total_amount=round(prediction, 2),
        model_name=str(MODEL_ARTIFACT.get("model_name", "unknown")),
    )
