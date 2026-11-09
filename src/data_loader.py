"""Data loading and validation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_TARGET_COLUMN = "SalePrice"


def load_csv(path: str | Path, dataset_name: str) -> pd.DataFrame:
    """Load a CSV file and validate that it exists and is non-empty."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"{dataset_name} file not found: {csv_path}")
    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"{dataset_name} must be a CSV file: {csv_path}")

    dataframe = pd.read_csv(csv_path)
    if dataframe.empty:
        raise ValueError(f"{dataset_name} is empty: {csv_path}")
    if dataframe.columns.duplicated().any():
        raise ValueError(f"{dataset_name} has duplicate column names: {csv_path}")
    return dataframe


def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: Iterable[str],
    dataset_name: str,
) -> None:
    """Ensure expected columns exist in a dataframe."""
    missing_columns = [col for col in required_columns if col not in dataframe.columns]
    if missing_columns:
        missing_str = ", ".join(missing_columns)
        raise ValueError(f"{dataset_name} is missing required columns: {missing_str}")


def load_train_test_data(
    train_path: str | Path,
    test_path: str | Path,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test datasets with consistent validation."""
    train_df = load_csv(train_path, "Training dataset")
    test_df = load_csv(test_path, "Test dataset")

    validate_required_columns(train_df, [target_column], "Training dataset")

    if target_column in test_df.columns:
        raise ValueError(
            f"Test dataset should not include target column '{target_column}' to avoid leakage."
        )

    return train_df, test_df
