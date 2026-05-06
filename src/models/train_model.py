"""Train and export the production taxi fare model."""

from __future__ import annotations

import argparse
import math
import time
from typing import Dict, Iterable, Tuple

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from src.data.ingestion import fetch_data_in_batches, fetch_dataframe, fetch_sample
from src.features.build_features import (
    LOW_CARDINALITY_CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    make_model_pipeline,
)
from src.utils.config import get_env, get_project_config


DEFAULT_TOP_N_LOCATION_PAIRS = 300
DEFAULT_TRAIN_SAMPLE_PCT = 0.004
DEFAULT_VAL_SAMPLE_PCT = 0.02
DEFAULT_TEST_SAMPLE_PCT = 0.02
DEFAULT_TRAIN_LIMIT = 50_000
DEFAULT_VAL_LIMIT = 15_000
DEFAULT_TEST_LIMIT = 15_000


def regression_metrics(y_true, y_pred) -> Tuple[float, float, float]:
    """Return RMSE, MAE and R2."""
    return (
        root_mean_squared_error(y_true, y_pred),
        mean_absolute_error(y_true, y_pred),
        r2_score(y_true, y_pred),
    )


def build_candidate_models(random_state: int) -> Dict[str, object]:
    """Create the baseline, ensembles and required boosting candidates."""
    return {
        "baseline_ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=80,
            max_depth=16,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=random_state,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=80,
            max_depth=18,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=random_state,
        ),
        "voting_rf_et_ridge": VotingRegressor(
            estimators=[
                (
                    "rf",
                    RandomForestRegressor(
                        n_estimators=50,
                        max_depth=14,
                        min_samples_leaf=25,
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
                (
                    "et",
                    ExtraTreesRegressor(
                        n_estimators=50,
                        max_depth=16,
                        min_samples_leaf=25,
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
                ("ridge", Ridge(alpha=1.0)),
            ],
            n_jobs=-1,
        ),
        "bagging_tree": BaggingRegressor(
            estimator=DecisionTreeRegressor(
                max_depth=16,
                min_samples_leaf=25,
                random_state=random_state,
            ),
            n_estimators=60,
            bootstrap=True,
            n_jobs=-1,
            random_state=random_state,
        ),
        "pasting_tree": BaggingRegressor(
            estimator=DecisionTreeRegressor(
                max_depth=16,
                min_samples_leaf=25,
                random_state=random_state,
            ),
            n_estimators=60,
            bootstrap=False,
            n_jobs=-1,
            random_state=random_state,
        ),
        "adaboost": AdaBoostRegressor(
            estimator=DecisionTreeRegressor(
                max_depth=8,
                min_samples_leaf=30,
                random_state=random_state,
            ),
            n_estimators=80,
            learning_rate=0.05,
            random_state=random_state,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=160,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=random_state,
        ),
        "xgboost": XGBRegressor(
            n_estimators=220,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=260,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        ),
        "catboost": CatBoostRegressor(
            iterations=220,
            learning_rate=0.05,
            depth=7,
            loss_function="RMSE",
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
        ),
    }


def get_top_location_pairs(train_table: str, top_n: int) -> Iterable[str]:
    """Read the most frequent train routes from Snowflake."""
    top_df = fetch_dataframe(
        f"""
        SELECT LOCATION_PAIR, COUNT(*) AS ROWS_TOTAL
        FROM {train_table}
        GROUP BY 1
        ORDER BY ROWS_TOTAL DESC
        LIMIT {int(top_n)}
        """
    )
    return top_df["LOCATION_PAIR"].astype(str).tolist()


def load_training_samples(
    train_table: str,
    val_table: str,
    test_table: str,
    target_column: str,
    train_sample_pct: float,
    val_sample_pct: float,
    test_sample_pct: float,
    train_limit: int,
    val_limit: int,
    test_limit: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load bounded temporal samples from Snowflake."""
    columns = MODEL_FEATURES + [target_column]
    train_df = fetch_sample(train_table, columns, train_sample_pct, train_limit)
    val_df = fetch_sample(val_table, columns, val_sample_pct, val_limit)
    test_df = fetch_sample(test_table, columns, test_sample_pct, test_limit)
    return train_df, val_df, test_df


def evaluate_pipeline(name: str, pipeline, X_train, y_train, X_val, y_val) -> dict:
    """Fit a pipeline and return train/validation metrics."""
    started = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_pred = pipeline.predict(X_train)
    val_pred = pipeline.predict(X_val)
    elapsed = time.perf_counter() - started

    train_rmse, train_mae, train_r2 = regression_metrics(y_train, train_pred)
    val_rmse, val_mae, val_r2 = regression_metrics(y_val, val_pred)
    return {
        "model": name,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "train_mae": train_mae,
        "val_mae": val_mae,
        "train_r2": train_r2,
        "val_r2": val_r2,
        "fit_seconds": elapsed,
    }


def evaluate_pipeline_in_batches(
    pipeline,
    query: str,
    target_column: str,
    batch_size: int,
    max_batches: int | None = None,
) -> dict:
    """Evaluate a fitted pipeline over Snowflake batches without loading all rows."""
    rows_total = 0
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    y_sum = 0.0
    y_square_sum = 0.0
    batches_seen = 0

    for batch in fetch_data_in_batches(query, batch_size=batch_size):
        if batch.empty:
            continue
        X_batch = batch[MODEL_FEATURES]
        y_batch = pd.to_numeric(batch[target_column], errors="coerce")
        valid_mask = y_batch.notna()
        if not valid_mask.any():
            continue

        y_batch = y_batch.loc[valid_mask]
        X_batch = X_batch.loc[valid_mask]
        predictions = pipeline.predict(X_batch)
        errors = y_batch.to_numpy() - predictions

        rows_total += len(y_batch)
        squared_error_sum += float((errors**2).sum())
        absolute_error_sum += float(abs(errors).sum())
        y_sum += float(y_batch.sum())
        y_square_sum += float((y_batch**2).sum())
        batches_seen += 1

        if max_batches is not None and batches_seen >= max_batches:
            break

    if rows_total == 0:
        raise ValueError("No rows were evaluated in batch mode.")

    total_variance_sum = y_square_sum - (y_sum**2 / rows_total)
    r2 = 1 - squared_error_sum / total_variance_sum if total_variance_sum > 0 else math.nan
    return {
        "rows": rows_total,
        "batches": batches_seen,
        "rmse": math.sqrt(squared_error_sum / rows_total),
        "mae": absolute_error_sum / rows_total,
        "r2": r2,
    }


def train_and_export(
    model_name: str = "gradient_boosting",
    compare_all: bool = False,
    batch_test: bool = False,
    max_test_batches: int | None = None,
) -> dict:
    """Train model candidates from Snowflake samples and export the winner."""
    cfg = get_project_config(required=True)
    train_table = cfg.fq_table(cfg.train_table)
    val_table = cfg.fq_table(cfg.val_table)
    test_table = cfg.fq_table(cfg.test_table)
    target_column = cfg.target_column or TARGET_COLUMN

    top_n = int(get_env("TOP_N_LOCATION_PAIRS", str(DEFAULT_TOP_N_LOCATION_PAIRS)))
    train_sample_pct = float(get_env("TRAIN_SAMPLE_PCT", str(DEFAULT_TRAIN_SAMPLE_PCT)))
    val_sample_pct = float(get_env("VAL_SAMPLE_PCT", str(DEFAULT_VAL_SAMPLE_PCT)))
    test_sample_pct = float(get_env("TEST_SAMPLE_PCT", str(DEFAULT_TEST_SAMPLE_PCT)))
    train_limit = int(get_env("TRAIN_LIMIT", str(DEFAULT_TRAIN_LIMIT)))
    val_limit = int(get_env("VAL_LIMIT", str(DEFAULT_VAL_LIMIT)))
    test_limit = int(get_env("TEST_LIMIT", str(DEFAULT_TEST_LIMIT)))

    top_location_pairs = list(get_top_location_pairs(train_table, top_n))
    train_df, val_df, test_df = load_training_samples(
        train_table,
        val_table,
        test_table,
        target_column,
        train_sample_pct,
        val_sample_pct,
        test_sample_pct,
        train_limit,
        val_limit,
        test_limit,
    )

    X_train = train_df[MODEL_FEATURES]
    y_train = train_df[target_column]
    X_val = val_df[MODEL_FEATURES]
    y_val = val_df[target_column]

    candidates = build_candidate_models(cfg.random_state)
    selected_names = list(candidates) if compare_all else [model_name]
    fitted_pipelines = {}
    results = []

    for name in selected_names:
        if name not in candidates:
            raise ValueError(f"Unknown model '{name}'. Available: {sorted(candidates)}")
        print(f"Training {name}...")
        pipeline = make_model_pipeline(clone(candidates[name]), top_location_pairs)
        metrics = evaluate_pipeline(name, pipeline, X_train, y_train, X_val, y_val)
        fitted_pipelines[name] = pipeline
        results.append(metrics)
        print(f"{name}: val_rmse={metrics['val_rmse']:.4f}")

    results_df = pd.DataFrame(results).sort_values("val_rmse").reset_index(drop=True)
    best_model_name = str(results_df.iloc[0]["model"])
    best_pipeline = fitted_pipelines[best_model_name]

    X_test = test_df[MODEL_FEATURES]
    y_test = test_df[target_column]
    test_pred = best_pipeline.predict(X_test)
    test_rmse, test_mae, test_r2 = regression_metrics(y_test, test_pred)
    test_metrics_df = pd.DataFrame(
        [
            {
                "model": best_model_name,
                "test_rows": len(test_df),
                "test_rmse": test_rmse,
                "test_mae": test_mae,
                "test_r2": test_r2,
            }
        ]
    )

    batch_test_metrics = None
    if batch_test:
        batch_query = f"SELECT {', '.join(MODEL_FEATURES + [target_column])} FROM {test_table}"
        batch_test_metrics = evaluate_pipeline_in_batches(
            best_pipeline,
            batch_query,
            target_column,
            batch_size=cfg.batch_size,
            max_batches=max_test_batches,
        )
        print("Batch test metrics:", batch_test_metrics)

    artifact = {
        "model_name": best_model_name,
        "pipeline": best_pipeline,
        "model_features": MODEL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": LOW_CARDINALITY_CATEGORICAL_FEATURES,
        "top_location_pairs": sorted(top_location_pairs),
        "validation_results": results_df,
        "test_metrics": test_metrics_df,
        "batch_test_metrics": batch_test_metrics,
    }

    cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, cfg.model_path)
    print(f"Model exported to: {cfg.model_path}")
    print(test_metrics_df)
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and export the taxi fare model.")
    parser.add_argument("--model", default="gradient_boosting", help="Candidate model to train.")
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Train all candidate models and export the best validation model.",
    )
    parser.add_argument(
        "--batch-test",
        action="store_true",
        help="Evaluate the exported winner over TEST_SET with Snowflake batches.",
    )
    parser.add_argument(
        "--max-test-batches",
        type=int,
        default=None,
        help="Optional cap for batch test evaluation during demos or smoke checks.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_export(
        model_name=args.model,
        compare_all=args.compare_all,
        batch_test=args.batch_test,
        max_test_batches=args.max_test_batches,
    )
