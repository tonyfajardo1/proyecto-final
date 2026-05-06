"""Streamlit frontend for the taxi fare prediction API."""

from __future__ import annotations

from datetime import date, time
from pathlib import Path
import sys

import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.config import get_project_config


def build_payload(values: dict) -> dict:
    return {
        "pickup_date": values["pickup_date"].isoformat(),
        "pickup_time": values["pickup_time"].strftime("%H:%M:%S"),
        "trip_distance": values["trip_distance"],
        "passenger_count": values["passenger_count"],
        "service_type": values["service_type"],
        "vendor_id": str(values["vendor_id"]),
        "rate_code_id": str(values["rate_code_id"]),
        "trip_type": str(values["trip_type"]),
        "pu_borough": values["pu_borough"],
        "do_borough": values["do_borough"],
        "pu_location_id": values["pu_location_id"],
        "do_location_id": values["do_location_id"],
    }


cfg = get_project_config(required=False)
st.set_page_config(page_title="Prediccion de tarifa", page_icon=None, layout="centered")
st.title("Prediccion de tarifa NYC Taxi")

with st.sidebar:
    api_url = st.text_input("API URL", value=cfg.api_url.replace("/predict", "/predict-simple"))

boroughs = ["Manhattan", "Queens", "Brooklyn", "Bronx", "Staten Island", "EWR", "Unknown"]

with st.form("prediction_form"):
    left, right = st.columns(2)

    with left:
        pickup_date = st.date_input("Fecha", value=date(2025, 1, 15))
        pickup_time = st.time_input("Hora", value=time(12, 0))
        service_type = st.selectbox("Servicio", ["yellow", "green"])
        vendor_id = st.selectbox("Vendor ID", ["1", "2", "Unknown"])
        rate_code_id = st.selectbox("Rate code", ["1", "2", "3", "4", "5", "6", "Unknown"])
        trip_type = st.selectbox("Trip type", ["1", "2", "Unknown"])

    with right:
        trip_distance = st.number_input("Distancia (millas)", min_value=0.1, max_value=100.0, value=2.5, step=0.1)
        passenger_count = st.number_input("Pasajeros", min_value=1, max_value=6, value=1, step=1)
        pu_location_id = st.number_input("Pickup location ID", min_value=1, max_value=999, value=237, step=1)
        do_location_id = st.number_input("Dropoff location ID", min_value=1, max_value=999, value=236, step=1)
        pu_borough = st.selectbox("Pickup borough", boroughs, index=0)
        do_borough = st.selectbox("Dropoff borough", boroughs, index=0)

    submitted = st.form_submit_button("Predecir tarifa", use_container_width=True)

if submitted:
    payload = build_payload(
        {
            "pickup_date": pickup_date,
            "pickup_time": pickup_time,
            "service_type": service_type,
            "vendor_id": vendor_id,
            "rate_code_id": rate_code_id,
            "trip_type": trip_type,
            "trip_distance": trip_distance,
            "passenger_count": passenger_count,
            "pu_location_id": pu_location_id,
            "do_location_id": do_location_id,
            "pu_borough": pu_borough,
            "do_borough": do_borough,
        }
    )

    try:
        response = requests.post(api_url, json=payload, timeout=15)
        response.raise_for_status()
        result = response.json()
        st.metric("Tarifa estimada", f"${result['estimated_total_amount']:,.2f}")
        st.caption(f"Modelo: {result.get('model_name', 'unknown')}")
    except requests.RequestException as exc:
        st.error(f"No se pudo consultar la API: {exc}")
