import pandas as pd
import pytest

from src.data_loader import load_csv, load_train_test_data, validate_required_columns


def test_load_csv_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "missing.csv", "Training dataset")


def test_validate_required_columns_raises_on_missing():
    dataframe = pd.DataFrame({"A": [1], "B": [2]})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_required_columns(dataframe, ["A", "C"], "Training dataset")


def test_load_train_test_data_validates_target(tmp_path):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"

    pd.DataFrame({"Id": [1], "Feature": [10], "SalePrice": [200000]}).to_csv(train_path, index=False)
    pd.DataFrame({"Id": [2], "Feature": [20]}).to_csv(test_path, index=False)

    train_df, test_df = load_train_test_data(train_path, test_path)

    assert "SalePrice" in train_df.columns
    assert "SalePrice" not in test_df.columns


def test_load_train_test_data_rejects_target_in_test(tmp_path):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"

    pd.DataFrame({"Id": [1], "SalePrice": [200000]}).to_csv(train_path, index=False)
    pd.DataFrame({"Id": [2], "SalePrice": [300000]}).to_csv(test_path, index=False)

    with pytest.raises(ValueError, match="should not include target"):
        load_train_test_data(train_path, test_path)
