"""Training utilities for CatBoost house price regression."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

from src.data_loader import DEFAULT_TARGET_COLUMN, load_train_test_data
from src.feature_engineering import add_engineered_features
from src.preprocessing import build_preprocessor


@dataclass(slots=True)
class TrainingConfig:
    """Configuration for model training and optional hyperparameter search."""

    train_path: str = "data/raw/train.csv"
    test_path: str = "data/raw/test.csv"
    model_output_path: str = "models/catboost_pipeline.joblib"
    target_column: str = DEFAULT_TARGET_COLUMN
    random_state: int = 42
    cv_splits: int = 5
    tune_hyperparameters: bool = True
    n_iter: int = 20
    log_target: bool = True


def build_catboost_model(random_state: int = 42) -> CatBoostRegressor:
    """Create base CatBoost regressor."""
    return CatBoostRegressor(
        loss_function="RMSE",
        random_seed=random_state,
        verbose=0,
    )


def build_training_pipeline(feature_df: pd.DataFrame, random_state: int = 42) -> Pipeline:
    """Create train/inference pipeline with preprocessing and CatBoost model."""
    preprocessor = build_preprocessor(feature_df)
    model = build_catboost_model(random_state=random_state)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def get_hyperparameter_space() -> dict[str, list[Any]]:
    """Readable CatBoost hyperparameter search space."""
    return {
        "model__depth": [4, 6, 8, 10],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__l2_leaf_reg": [1, 3, 5, 7, 9],
        "model__iterations": [300, 500, 800, 1200],
        "model__subsample": [0.7, 0.85, 1.0],
    }


def _target_transform(y: pd.Series, log_target: bool) -> pd.Series:
    return np.log1p(y) if log_target else y


def train_model(train_df: pd.DataFrame, config: TrainingConfig) -> dict[str, Any]:
    """Train CatBoost pipeline, optionally tuning hyperparameters."""
    if config.target_column not in train_df.columns:
        raise ValueError(f"Target column '{config.target_column}' not found in training data.")

    y = _target_transform(train_df[config.target_column], config.log_target)
    x = add_engineered_features(train_df.drop(columns=[config.target_column]))

    pipeline = build_training_pipeline(x, random_state=config.random_state)
    cv = KFold(n_splits=config.cv_splits, shuffle=True, random_state=config.random_state)

    if config.tune_hyperparameters:
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=get_hyperparameter_space(),
            n_iter=config.n_iter,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            random_state=config.random_state,
            n_jobs=-1,
            refit=True,
        )
        search.fit(x, y)
        fitted_pipeline: Pipeline = search.best_estimator_
        best_cv_rmse = float(-search.best_score_)
        best_params = search.best_params_
    else:
        pipeline.fit(x, y)
        fitted_pipeline = pipeline
        scores = cross_val_score(
            fitted_pipeline,
            x,
            y,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1,
        )
        best_cv_rmse = float(np.mean(-scores))
        best_params = {}

    return {
        "pipeline": fitted_pipeline,
        "target_column": config.target_column,
        "log_target": config.log_target,
        "best_cv_rmse": best_cv_rmse,
        "best_params": best_params,
    }


def save_model_bundle(model_bundle: dict[str, Any], output_path: str | Path) -> None:
    """Persist a fitted model bundle for prediction."""
    model_path = Path(output_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, model_path)


def run_training(config: TrainingConfig) -> dict[str, Any]:
    """Train model from configured file paths and save artifact."""
    train_df, _ = load_train_test_data(
        train_path=config.train_path,
        test_path=config.test_path,
        target_column=config.target_column,
    )
    model_bundle = train_model(train_df, config)
    save_model_bundle(model_bundle, config.model_output_path)
    return model_bundle
