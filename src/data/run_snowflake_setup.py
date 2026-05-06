import argparse
import re
import sys
from pathlib import Path

import snowflake.connector
from snowflake.connector.errors import Error as SnowflakeError

from src.utils.config import PROJECT_ROOT, get_project_config

SQL_DIR = PROJECT_ROOT / "src" / "data" / "sql"
SQL_FILES = [
    SQL_DIR / "01_create_obt.sql",
    SQL_DIR / "02_create_splits.sql",
]


def _strip_line_comments(sql: str) -> str:
    """Remove SQL line comments before splitting statements."""
    lines = []
    for line in sql.splitlines():
        stripped = line.strip()
        if stripped.startswith("--") or stripped == "":
            continue
        lines.append(line)
    return "\n".join(lines)


def _split_sql_statements(sql: str) -> list[str]:
    """Split project SQL files into executable statements."""
    sql_without_comments = _strip_line_comments(sql)
    return [
        statement.strip()
        for statement in sql_without_comments.split(";")
        if statement.strip()
    ]


def _schema_name(fq_name: str) -> str:
    parts = fq_name.split(".")
    return parts[0] if len(parts) > 1 else "PUBLIC"


def _object_name(fq_name: str) -> str:
    return fq_name.split(".")[-1]


def execute_sql_files(conn, sql_files: list[Path], dry_run: bool = False) -> None:
    cursor = None if dry_run else conn.cursor()
    try:
        for sql_file in sql_files:
            sql = sql_file.read_text(encoding="utf-8")
            statements = _split_sql_statements(sql)
            print(f"\n{sql_file.name}: {len(statements)} statement(s)")
            for index, statement in enumerate(statements, start=1):
                preview = re.sub(r"\s+", " ", statement).strip()[:120]
                print(f"  [{index}] {preview}")
                if not dry_run:
                    cursor.execute(statement)
    finally:
        if cursor is not None:
            cursor.close()


def validate_outputs(conn) -> None:
    cfg = get_project_config(required=True)
    cursor = conn.cursor()
    model_table = cfg.fq_table(cfg.obt_model_table)
    train_table = cfg.fq_table(cfg.train_table)
    val_table = cfg.fq_table(cfg.val_table)
    test_table = cfg.fq_table(cfg.test_table)

    validations = [
        (
            "model_table_summary",
            f"""
            SELECT
                COUNT(*) AS rows_model,
                MIN(YEAR) AS min_year,
                MAX(YEAR) AS max_year,
                ROUND(AVG({cfg.target_column}), 2) AS avg_target
            FROM {model_table}
            """,
        ),
        (
            "split_summary",
            f"""
            SELECT 'TRAIN_SET' AS split_name, COUNT(*) AS rows_split, MIN(YEAR) AS min_year, MAX(YEAR) AS max_year
            FROM {train_table}
            UNION ALL
            SELECT 'VAL_SET' AS split_name, COUNT(*) AS rows_split, MIN(YEAR) AS min_year, MAX(YEAR) AS max_year
            FROM {val_table}
            UNION ALL
            SELECT 'TEST_SET' AS split_name, COUNT(*) AS rows_split, MIN(YEAR) AS min_year, MAX(YEAR) AS max_year
            FROM {test_table}
            ORDER BY split_name
            """,
        ),
        (
            "leakage_column_check",
            f"""
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = '{_schema_name(model_table).upper()}'
              AND TABLE_NAME = '{_object_name(model_table).upper()}'
              AND COLUMN_NAME IN (
                  'DROPOFF_DATETIME', 'DROPOFF_DATE', 'DROPOFF_HOUR',
                  'FARE_AMOUNT', 'EXTRA', 'MTA_TAX', 'TIP_AMOUNT', 'TOLLS_AMOUNT',
                  'IMPROVEMENT_SURCHARGE', 'CONGESTION_SURCHARGE', 'AIRPORT_FEE',
                  'TRIP_DURATION_MIN', 'AVG_SPEED_MPH', 'TIP_PCT',
                  'PAYMENT_TYPE', 'PAYMENT_TYPE_DESC'
              )
            ORDER BY COLUMN_NAME
            """,
        ),
    ]

    try:
        for name, query in validations:
            print(f"\n{name}")
            cursor.execute(query)
            columns = [col[0] for col in cursor.description]
            print(" | ".join(columns))
            for row in cursor.fetchall():
                print(" | ".join("" if value is None else str(value) for value in row))
    finally:
        cursor.close()


def source_table_exists(conn) -> bool:
    cfg = get_project_config(required=True)
    cursor = conn.cursor()
    source_table = cfg.fq_table(cfg.obt_source_table)
    try:
        cursor.execute(
            f"""
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = '{_schema_name(source_table).upper()}'
              AND TABLE_NAME = '{_object_name(source_table).upper()}'
            """
        )
        return cursor.fetchone()[0] > 0
    finally:
        cursor.close()


def connect():
    cfg = get_project_config(required=True)
    return snowflake.connector.connect(**cfg.snowflake.connector_kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create modeling OBT, create splits, and validate Snowflake outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Print statements without executing them.")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation queries after execution.")
    args = parser.parse_args()

    if args.dry_run:
        execute_sql_files(None, SQL_FILES, dry_run=True)
        return 0

    try:
        conn = connect()
    except SnowflakeError as exc:
        print("Snowflake connection failed. Credentials were not printed.")
        print(str(exc))
        return 1

    try:
        if not source_table_exists(conn):
            cfg = get_project_config(required=True)
            source_table = cfg.fq_table(cfg.obt_source_table)
            print(f"Missing required source table: {source_table}")
            print("Recreate it first from the Deber 3 ingestion/OBT notebooks, or update OBT_SOURCE_TABLE/.env if it exists with another name.")
            return 1
        execute_sql_files(conn, SQL_FILES, dry_run=args.dry_run)
        if not args.dry_run and not args.skip_validation:
            validate_outputs(conn)
    except SnowflakeError as exc:
        print("Snowflake execution failed.")
        print(str(exc))
        return 1
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
