import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.feature_engineering import add_engineered_features
from src.predict import predict_from_dataframe
from src.preprocessing import build_preprocessor


def _training_frame():
    return pd.DataFrame(
        {
            "TotalBsmtSF": [800, 700, 900],
            "1stFlrSF": [900, 850, 920],
            "2ndFlrSF": [500, 400, 600],
            "YrSold": [2010, 2009, 2010],
            "YearBuilt": [2000, 1995, 2005],
            "YearRemodAdd": [2005, 2000, 2007],
            "GarageYrBlt": [2002, 1997, 2006],
            "FullBath": [2, 1, 2],
            "HalfBath": [1, 0, 1],
            "BsmtFullBath": [1, 1, 1],
            "BsmtHalfBath": [0, 0, 0],
            "Neighborhood": ["A", "B", "A"],
        }
    )


def test_predict_from_dataframe_returns_series():
    features = _training_frame()
    target = pd.Series([200000, 180000, 230000])

    engineered = add_engineered_features(features)
    preprocessor = build_preprocessor(engineered)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", LinearRegression())])
    pipeline.fit(engineered, target)

    predictions = predict_from_dataframe({"pipeline": pipeline, "log_target": False}, features)

    assert list(predictions.index) == [0, 1, 2]
    assert predictions.name == "PredictedSalePrice"


def test_predict_from_dataframe_reverses_log_target():
    features = _training_frame()
    target_log = np.log1p(pd.Series([200000, 180000, 230000]))

    engineered = add_engineered_features(features)
    preprocessor = build_preprocessor(engineered)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", LinearRegression())])
    pipeline.fit(engineered, target_log)

    predictions = predict_from_dataframe({"pipeline": pipeline, "log_target": True}, features)

    assert (predictions > 0).all()
