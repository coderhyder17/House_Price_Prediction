"""Preprocessing pipelines for numeric and categorical features."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_DROP_COLUMNS = ["Id"]


def split_feature_types(
    dataframe: pd.DataFrame,
    drop_columns: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Split feature names by numerical and categorical dtypes."""
    drop_columns = drop_columns or []
    feature_df = dataframe.drop(columns=drop_columns, errors="ignore")

    numerical_features = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = feature_df.select_dtypes(exclude=["number"]).columns.tolist()
    return numerical_features, categorical_features


def build_preprocessor(
    dataframe: pd.DataFrame,
    drop_columns: list[str] | None = None,
) -> ColumnTransformer:
    """Create a ColumnTransformer consistent for training and inference."""
    drop_columns = drop_columns if drop_columns is not None else DEFAULT_DROP_COLUMNS
    numerical_features, categorical_features = split_feature_types(dataframe, drop_columns)

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
