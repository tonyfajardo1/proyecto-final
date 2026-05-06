"""Snowflake access helpers for sampled and batch data extraction."""

from __future__ import annotations

from typing import Iterator, Optional, Sequence

import pandas as pd
import snowflake.connector

from src.utils.config import get_project_config, get_snowflake_config


def get_snowflake_connection():
    """Create a Snowflake connector connection from the project .env file."""
    cfg = get_snowflake_config(required=True)
    return snowflake.connector.connect(**cfg.connector_kwargs)


def fetch_dataframe(query: str) -> pd.DataFrame:
    """Execute a SQL query and return a pandas DataFrame."""
    with get_snowflake_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            return cur.fetch_pandas_all()


def fetch_data_in_batches(
    query: str,
    batch_size: Optional[int] = None,
) -> Iterator[pd.DataFrame]:
    """Yield query results as pandas DataFrames without loading all rows at once."""
    cfg = get_project_config(required=True)
    selected_batch_size = batch_size or cfg.batch_size

    with get_snowflake_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)

            try:
                for batch in cur.fetch_pandas_batches():
                    if not batch.empty:
                        yield batch
                return
            except Exception:
                pass

            column_names = [column[0] for column in cur.description]
            while True:
                rows = cur.fetchmany(selected_batch_size)
                if not rows:
                    break
                yield pd.DataFrame.from_records(rows, columns=column_names)


def fetch_sample(
    table_name: str,
    columns: Optional[Sequence[str]] = None,
    sample_pct: Optional[float] = None,
    limit: Optional[int] = None,
    where: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch a bounded table sample for notebooks or local model training."""
    selected_columns = ", ".join(columns) if columns else "*"
    sample_clause = f" SAMPLE BERNOULLI ({sample_pct})" if sample_pct else ""
    where_clause = f" WHERE {where}" if where else ""
    limit_clause = f" LIMIT {int(limit)}" if limit else ""
    query = f"SELECT {selected_columns} FROM {table_name}{sample_clause}{where_clause}{limit_clause}"
    return fetch_dataframe(query)


def fetch_table_count(table_name: str) -> int:
    """Return row count for a Snowflake table or view."""
    result = fetch_dataframe(f"SELECT COUNT(*) AS ROWS_TOTAL FROM {table_name}")
    return int(result.iloc[0]["ROWS_TOTAL"])
