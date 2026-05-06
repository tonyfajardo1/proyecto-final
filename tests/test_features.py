import pandas as pd

from src.features.build_features import TripFeatureEngineer


def test_trip_feature_engineer_builds_route_features():
    frame = pd.DataFrame(
        [
            {
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
        ]
    )

    transformed = TripFeatureEngineer(top_location_pairs={"237_236"}).transform(frame)

    assert "LOCATION_PAIR" not in transformed.columns
    assert transformed.loc[0, "LOCATION_PAIR_TOP"] == "237_236"
    assert transformed.loc[0, "PU_LOCATION_ID"] == "237"
    assert transformed.loc[0, "DO_LOCATION_ID"] == "236"
