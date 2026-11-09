"""Model evaluation and feature ablation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.feature_engineering import ENGINEERED_FEATURES, add_engineered_features
from src.preprocessing import build_preprocessor


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate core regression metrics for house price prediction."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    if np.any(y_true < 0) or np.any(y_pred < 0):
        rmsle = float("nan")
    else:
        rmsle = float(
            np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(np.maximum(y_pred, 0))))
        )

    return {"rmse": rmse, "mae": mae, "r2": r2, "rmsle": rmsle}


def _build_pipeline(dataframe: pd.DataFrame, estimator: Any) -> Pipeline:
    preprocessor: ColumnTransformer = build_preprocessor(dataframe)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])


def run_feature_ablation(
    dataframe: pd.DataFrame,
    target_column: str,
    estimator: Any,
    random_state: int = 42,
    cv_splits: int = 5,
) -> pd.DataFrame:
    """Evaluate full features and one-at-a-time engineered feature ablation."""
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    y = dataframe[target_column]
    x_base = add_engineered_features(dataframe.drop(columns=[target_column]))

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    scenarios = [("All engineered features", None)]
    scenarios.extend((f"Without {feature}", feature) for feature in ENGINEERED_FEATURES)

    rows: list[dict[str, float | str]] = []
    for scenario_name, removed_feature in scenarios:
        x_scenario = x_base.drop(columns=[removed_feature], errors="ignore")
        pipeline = _build_pipeline(x_scenario, estimator)

        scores = cross_val_score(
            pipeline,
            x_scenario,
            y,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1,
        )
        rmse_scores = -scores

        rows.append(
            {
                "scenario": scenario_name,
                "rmse_mean": float(np.mean(rmse_scores)),
                "rmse_std": float(np.std(rmse_scores)),
            }
        )

    return pd.DataFrame(rows).sort_values(by="rmse_mean").reset_index(drop=True)
