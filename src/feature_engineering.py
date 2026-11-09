"""Feature engineering utilities for Ames Housing."""

from __future__ import annotations

import numpy as np
import pandas as pd

ENGINEERED_FEATURES = ["TotalSF", "HouseAge", "RemodAge", "GarageAge", "TotalBathrooms"]


def _require_columns(dataframe: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing columns for feature engineering: {', '.join(missing)}")


def _clip_age(values: pd.Series) -> pd.Series:
    return values.fillna(0).clip(lower=0)


def add_engineered_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features used in the notebook workflow."""
    required = [
        "TotalBsmtSF",
        "1stFlrSF",
        "2ndFlrSF",
        "YrSold",
        "YearBuilt",
        "YearRemodAdd",
        "GarageYrBlt",
        "FullBath",
        "HalfBath",
        "BsmtFullBath",
        "BsmtHalfBath",
    ]
    _require_columns(dataframe, required)

    df = dataframe.copy()

    df["TotalSF"] = (
        df["TotalBsmtSF"].fillna(0) + df["1stFlrSF"].fillna(0) + df["2ndFlrSF"].fillna(0)
    )

    year_sold = df["YrSold"]
    df["HouseAge"] = _clip_age(year_sold - df["YearBuilt"])
    df["RemodAge"] = _clip_age(year_sold - df["YearRemodAdd"])
    garage_year = df["GarageYrBlt"].fillna(df["YearBuilt"])
    df["GarageAge"] = _clip_age(year_sold - garage_year)

    df["TotalBathrooms"] = (
        df["FullBath"].fillna(0)
        + 0.5 * df["HalfBath"].fillna(0)
        + df["BsmtFullBath"].fillna(0)
        + 0.5 * df["BsmtHalfBath"].fillna(0)
    )

    return df
