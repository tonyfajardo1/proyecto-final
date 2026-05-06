-- ============================================================
-- 01_create_obt.sql
-- Project Final - Modeling OBT without target leakage
-- ============================================================
-- Run this script in Snowflake after ANALYTICS.OBT_TRIPS exists.
-- The ANALYTICS schema must already exist and your role must have
-- USAGE and CREATE TABLE permissions on it.
--
-- Goal:
--   Create ANALYTICS.OBT_TRIPS_MODEL with only features that can be
--   known before or at the beginning of the trip, plus the target
--   TOTAL_AMOUNT.
--
-- Excluded from model inputs because they are post-trip/leakage fields:
--   DROPOFF_DATETIME, DROPOFF_DATE, DROPOFF_HOUR, FARE_AMOUNT, EXTRA,
--   MTA_TAX, TIP_AMOUNT, TOLLS_AMOUNT, IMPROVEMENT_SURCHARGE,
--   CONGESTION_SURCHARGE, AIRPORT_FEE, TRIP_DURATION_MIN, AVG_SPEED_MPH,
--   TIP_PCT, PAYMENT_TYPE, PAYMENT_TYPE_DESC, lineage/audit columns.
--
-- If your group uses a different analytics schema, replace ANALYTICS
-- with the value from SNOWFLAKE_SCHEMA_ANALYTICS in .env.

CREATE OR REPLACE TABLE ANALYTICS.OBT_TRIPS_MODEL
CLUSTER BY (YEAR, MONTH, SERVICE_TYPE)
AS
SELECT
    -- Row identifier for traceability. Do not use as a model feature.
    TRIP_NK,

    -- Time features available at pickup.
    PICKUP_DATETIME,
    PICKUP_DATE,
    PICKUP_HOUR,
    DAY_OF_WEEK,
    MONTH,
    YEAR,
    CASE WHEN DAY_OF_WEEK IN (1, 7) THEN 1 ELSE 0 END AS IS_WEEKEND,
    CASE
        WHEN PICKUP_HOUR BETWEEN 6 AND 9 THEN 'morning_peak'
        WHEN PICKUP_HOUR BETWEEN 10 AND 15 THEN 'midday'
        WHEN PICKUP_HOUR BETWEEN 16 AND 20 THEN 'evening_peak'
        WHEN PICKUP_HOUR BETWEEN 21 AND 23 THEN 'night'
        ELSE 'overnight'
    END AS PICKUP_TIME_BAND,

    -- Spatial features. Taxi zone names help EDA; IDs are safer for modeling/API.
    PU_LOCATION_ID,
    PU_ZONE,
    PU_BOROUGH,
    DO_LOCATION_ID,
    DO_ZONE,
    DO_BOROUGH,
    CASE
        WHEN PU_BOROUGH = DO_BOROUGH THEN 1
        ELSE 0
    END AS SAME_BOROUGH_FLAG,
    CASE
        WHEN PU_LOCATION_ID IN (1, 132, 138) OR DO_LOCATION_ID IN (1, 132, 138)
            THEN 1
        ELSE 0
    END AS AIRPORT_TRIP_FLAG,
    CONCAT(TO_VARCHAR(PU_LOCATION_ID), '_', TO_VARCHAR(DO_LOCATION_ID)) AS LOCATION_PAIR,

    -- Service/catalog fields known before the fare is closed.
    SERVICE_TYPE,
    VENDOR_ID,
    VENDOR_NAME,
    RATE_CODE_ID,
    RATE_CODE_DESC,
    TRIP_TYPE,
    TRIP_TYPE_DESC,

    -- Trip request fields.
    PASSENGER_COUNT,
    TRIP_DISTANCE,

    -- Prediction target.
    TOTAL_AMOUNT
FROM ANALYTICS.OBT_TRIPS
WHERE
    -- Time coverage required by the project.
    YEAR BETWEEN 2015 AND 2025
    AND PICKUP_DATETIME IS NOT NULL
    AND PICKUP_DATE IS NOT NULL
    AND PICKUP_HOUR BETWEEN 0 AND 23
    AND DAY_OF_WEEK BETWEEN 1 AND 7

    -- Essential route/request fields.
    AND PU_LOCATION_ID IS NOT NULL
    AND DO_LOCATION_ID IS NOT NULL
    AND SERVICE_TYPE IN ('yellow', 'green')
    AND TRIP_DISTANCE IS NOT NULL
    AND TRIP_DISTANCE > 0
    AND TRIP_DISTANCE <= 100
    AND (PASSENGER_COUNT IS NULL OR PASSENGER_COUNT BETWEEN 1 AND 6)

    -- Target cleaning. Keep extreme but plausible trips, remove invalid tickets.
    AND TOTAL_AMOUNT IS NOT NULL
    AND TOTAL_AMOUNT > 0
    AND TOTAL_AMOUNT <= 500;

-- Quick validation queries to run after creation:
-- SELECT COUNT(*) AS rows_model, MIN(YEAR) AS min_year, MAX(YEAR) AS max_year
-- FROM ANALYTICS.OBT_TRIPS_MODEL;
--
-- SELECT YEAR, COUNT(*) AS rows_by_year, ROUND(AVG(TOTAL_AMOUNT), 2) AS avg_total_amount
-- FROM ANALYTICS.OBT_TRIPS_MODEL
-- GROUP BY 1
-- ORDER BY 1;
