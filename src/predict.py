"""Prediction utilities for trained house-price model pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.feature_engineering import add_engineered_features


def load_model_bundle(model_path: str | Path) -> dict[str, Any]:
    """Load saved model bundle from disk."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    bundle = joblib.load(path)

    if "pipeline" not in bundle:
        raise ValueError("Invalid model bundle: missing 'pipeline'.")
    return bundle


def predict_from_dataframe(model_bundle: dict[str, Any], features_df: pd.DataFrame) -> pd.Series:
    """Generate predictions from raw input rows using training-consistent processing."""
    engineered = add_engineered_features(features_df)
    raw_predictions = model_bundle["pipeline"].predict(engineered)

    if model_bundle.get("log_target", False):
        raw_predictions = np.expm1(raw_predictions)

    return pd.Series(raw_predictions, name="PredictedSalePrice")


def predict_from_csv(
    model_path: str | Path,
    input_csv_path: str | Path,
    output_csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load new data and generate price predictions."""
    model_bundle = load_model_bundle(model_path)
    input_df = pd.read_csv(input_csv_path)
    predictions = predict_from_dataframe(model_bundle, input_df)

    result = input_df.copy()
    result["PredictedSalePrice"] = predictions

    if output_csv_path is not None:
        output_path = Path(output_csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)

    return result
