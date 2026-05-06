from src.api import main


def test_predict_price_endpoint_function_returns_prediction():
    main.load_artifacts()
    trip = main.TripInput(
        trip_distance=2.5,
        passenger_count=1,
        pickup_hour=12,
        day_of_week=3,
        month=1,
        year=2025,
        is_weekend=0,
        same_borough_flag=1,
        airport_trip_flag=0,
        service_type="yellow",
        vendor_id="2",
        rate_code_id="1",
        trip_type="1",
        pickup_time_band="midday",
        pu_borough="Manhattan",
        do_borough="Manhattan",
        pu_location_id=237,
        do_location_id=236,
    )

    result = main.predict_price(trip)

    assert result.model_name == "gradient_boosting"
    assert result.estimated_total_amount > 0


def test_root_points_to_demo_endpoint():
    result = main.root()

    assert result["demo_endpoint"] == "/predict-simple"
    assert result["technical_endpoint"] == "/predict"


def test_predict_price_simple_endpoint_function_returns_prediction():
    main.load_artifacts()
    trip = main.TripSimpleInput(
        pickup_date="2025-01-15",
        pickup_time="12:00:00",
        trip_distance=2.5,
        passenger_count=1,
        service_type="yellow",
        vendor_id="2",
        rate_code_id="1",
        trip_type="1",
        pu_borough="Manhattan",
        do_borough="Manhattan",
        pu_location_id=237,
        do_location_id=236,
    )

    result = main.predict_price_simple(trip)

    assert result.model_name == "gradient_boosting"
    assert result.estimated_total_amount > 0
