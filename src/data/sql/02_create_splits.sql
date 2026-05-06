-- ============================================================
-- 02_create_splits.sql
-- Project Final - Time-based train/validation/test splits
-- ============================================================
-- These splits are created in Snowflake. Do not use train_test_split
-- locally, because that would mix time periods and can create leakage.
--
-- Required split:
--   TRAIN_SET: 2015-2023
--   VAL_SET:   2024
--   TEST_SET:  2025
--
-- If your group uses a different analytics schema, replace ANALYTICS
-- with the value from SNOWFLAKE_SCHEMA_ANALYTICS in .env.

CREATE OR REPLACE VIEW ANALYTICS.TRAIN_SET AS
SELECT *
FROM ANALYTICS.OBT_TRIPS_MODEL
WHERE YEAR BETWEEN 2015 AND 2023;

CREATE OR REPLACE VIEW ANALYTICS.VAL_SET AS
SELECT *
FROM ANALYTICS.OBT_TRIPS_MODEL
WHERE YEAR = 2024;

CREATE OR REPLACE VIEW ANALYTICS.TEST_SET AS
SELECT *
FROM ANALYTICS.OBT_TRIPS_MODEL
WHERE YEAR = 2025;

-- Quick validation queries to run after creation:
-- SELECT 'TRAIN_SET' AS split_name, COUNT(*) AS rows_split, MIN(YEAR) AS min_year, MAX(YEAR) AS max_year
-- FROM ANALYTICS.TRAIN_SET
-- UNION ALL
-- SELECT 'VAL_SET' AS split_name, COUNT(*) AS rows_split, MIN(YEAR) AS min_year, MAX(YEAR) AS max_year
-- FROM ANALYTICS.VAL_SET
-- UNION ALL
-- SELECT 'TEST_SET' AS split_name, COUNT(*) AS rows_split, MIN(YEAR) AS min_year, MAX(YEAR) AS max_year
-- FROM ANALYTICS.TEST_SET
-- ORDER BY split_name;
