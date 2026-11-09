import pandas as pd

from src.feature_engineering import add_engineered_features
from src.preprocessing import build_preprocessor


def _build_dataset():
    return pd.DataFrame(
        {
            "Id": [1, 2, 3],
            "Neighborhood": ["CollgCr", "OldTown", None],
            "TotalBsmtSF": [800, None, 700],
            "1stFlrSF": [900, 850, 780],
            "2ndFlrSF": [500, 0, 400],
            "YrSold": [2010, 2009, 2010],
            "YearBuilt": [2000, 1980, 1995],
            "YearRemodAdd": [2005, 1985, 2000],
            "GarageYrBlt": [2002, None, 1996],
            "FullBath": [2, 1, 2],
            "HalfBath": [1, 0, 1],
            "BsmtFullBath": [1, 0, 1],
            "BsmtHalfBath": [0, 1, 0],
        }
    )


def test_build_preprocessor_fit_transform():
    features = add_engineered_features(_build_dataset())
    preprocessor = build_preprocessor(features)
    transformed = preprocessor.fit_transform(features)

    assert transformed.shape[0] == 3
    assert transformed.shape[1] > 0


def test_build_preprocessor_handles_unseen_category():
    train_features = add_engineered_features(_build_dataset())
    preprocessor = build_preprocessor(train_features)
    preprocessor.fit(train_features)

    inference = train_features.copy()
    inference.loc[0, "Neighborhood"] = "NewPlace"
    transformed = preprocessor.transform(inference)

    assert transformed.shape[0] == 3
