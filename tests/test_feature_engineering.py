import numpy as np
import pandas as pd

from src.feature_engineering import add_engineered_features


def test_add_engineered_features_values():
    dataframe = pd.DataFrame(
        {
            "TotalBsmtSF": [800],
            "1stFlrSF": [900],
            "2ndFlrSF": [600],
            "YrSold": [2010],
            "YearBuilt": [2000],
            "YearRemodAdd": [2005],
            "GarageYrBlt": [2002],
            "FullBath": [2],
            "HalfBath": [1],
            "BsmtFullBath": [1],
            "BsmtHalfBath": [0],
        }
    )

    engineered = add_engineered_features(dataframe)

    assert engineered.loc[0, "TotalSF"] == 2300
    assert engineered.loc[0, "HouseAge"] == 10
    assert engineered.loc[0, "RemodAge"] == 5
    assert engineered.loc[0, "GarageAge"] == 8
    assert engineered.loc[0, "TotalBathrooms"] == 3.5


def test_add_engineered_features_handles_missing_garage_year():
    dataframe = pd.DataFrame(
        {
            "TotalBsmtSF": [500],
            "1stFlrSF": [700],
            "2ndFlrSF": [0],
            "YrSold": [2008],
            "YearBuilt": [1998],
            "YearRemodAdd": [2000],
            "GarageYrBlt": [np.nan],
            "FullBath": [1],
            "HalfBath": [0],
            "BsmtFullBath": [1],
            "BsmtHalfBath": [1],
        }
    )

    engineered = add_engineered_features(dataframe)
    assert engineered.loc[0, "GarageAge"] == 10
