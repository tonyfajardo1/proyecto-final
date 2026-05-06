import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def get_env(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """Read an environment variable with optional validation."""
    value = os.getenv(name, default)
    if required and (value is None or str(value).strip() == ""):
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def get_int_env(name: str, default: int) -> int:
    """Read an integer environment variable."""
    raw_value = get_env(name, str(default))
    try:
        return int(str(raw_value))
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer.") from exc


def resolve_project_path(path_value: Union[str, Path]) -> Path:
    """Resolve relative project paths from the repository root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


@dataclass(frozen=True)
class SnowflakeConfig:
    user: str
    password: str
    account: str
    warehouse: str
    database: str
    role: Optional[str] = None
    schema_raw: str = "RAW"
    schema_analytics: str = "ANALYTICS"
    host: Optional[str] = None
    port: str = "443"

    @property
    def connector_kwargs(self) -> dict:
        """Arguments expected by snowflake.connector.connect."""
        kwargs = {
            "user": self.user,
            "password": self.password,
            "account": self.account,
            "warehouse": self.warehouse,
            "database": self.database,
            "schema": self.schema_analytics,
        }
        if self.role:
            kwargs["role"] = self.role
        return kwargs

    @property
    def safe_preview(self) -> dict:
        """Connection preview without secrets, useful in notebooks."""
        return {
            "user": self.user,
            "account": self.account,
            "warehouse": self.warehouse,
            "database": self.database,
            "role": self.role,
            "schema_raw": self.schema_raw,
            "schema_analytics": self.schema_analytics,
            "host": self.host,
            "port": self.port,
        }


@dataclass(frozen=True)
class ProjectConfig:
    snowflake: SnowflakeConfig
    obt_source_table: str
    obt_model_table: str
    train_table: str
    val_table: str
    test_table: str
    target_column: str
    batch_size: int
    random_state: int
    model_path: Path
    api_url: str

    @property
    def analytics_schema(self) -> str:
        return self.snowflake.schema_analytics

    @property
    def raw_schema(self) -> str:
        return self.snowflake.schema_raw

    def fq_table(self, table_name: str, schema: Optional[str] = None) -> str:
        """Return a fully qualified Snowflake table/view name."""
        selected_schema = schema or self.analytics_schema
        return f"{selected_schema}.{table_name}"


def get_snowflake_config(required: bool = True) -> SnowflakeConfig:
    """Return Snowflake configuration from environment variables."""
    optional_default = None if required else ""
    return SnowflakeConfig(
        user=str(get_env("SNOWFLAKE_USER", optional_default, required=required)),
        password=str(get_env("SNOWFLAKE_PASSWORD", optional_default, required=required)),
        account=str(get_env("SNOWFLAKE_ACCOUNT", optional_default, required=required)),
        warehouse=str(get_env("SNOWFLAKE_WAREHOUSE", optional_default, required=required)),
        database=str(get_env("SNOWFLAKE_DATABASE", optional_default, required=required)),
        role=get_env("SNOWFLAKE_ROLE"),
        schema_raw=str(get_env("SNOWFLAKE_SCHEMA_RAW", "RAW")),
        schema_analytics=str(get_env("SNOWFLAKE_SCHEMA_ANALYTICS", "ANALYTICS")),
        host=get_env("SNOWFLAKE_HOST"),
        port=str(get_env("SNOWFLAKE_PORT", "443")),
    )


def get_project_config(required: bool = True) -> ProjectConfig:
    """Return all project settings from environment variables."""
    return ProjectConfig(
        snowflake=get_snowflake_config(required=required),
        obt_source_table=str(get_env("OBT_SOURCE_TABLE", "OBT_TRIPS")),
        obt_model_table=str(get_env("OBT_MODEL_TABLE", "OBT_TRIPS_MODEL")),
        train_table=str(get_env("TRAIN_TABLE", "TRAIN_SET")),
        val_table=str(get_env("VAL_TABLE", "VAL_SET")),
        test_table=str(get_env("TEST_TABLE", "TEST_SET")),
        target_column=str(get_env("TARGET_COLUMN", "TOTAL_AMOUNT")),
        batch_size=get_int_env("BATCH_SIZE", 100000),
        random_state=get_int_env("RANDOM_STATE", 42),
        model_path=resolve_project_path(str(get_env("MODEL_PATH", "data/processed/price_model.pkl"))),
        api_url=str(get_env("API_URL", "http://127.0.0.1:8000/predict")),
    )


def get_snowflake_credentials() -> dict:
    """Backward-compatible credential dictionary for older project modules."""
    return get_snowflake_config(required=True).connector_kwargs
