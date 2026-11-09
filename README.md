# House Price Prediction (Ames Housing)

A professional, reproducible machine learning project for predicting residential sale prices from the Ames Housing dataset using CatBoost and sklearn pipelines.

## Project Overview
This project predicts `SalePrice` for houses using a workflow that includes EDA, preprocessing, feature engineering, CatBoost regression, hyperparameter tuning, evaluation, and feature ablation.

## Problem Statement
Build an end-to-end regression pipeline that preserves the original methodology and produces maintainable training, evaluation, and prediction code.

## Dataset
- **Source:** Kaggle - *House Prices: Advanced Regression Techniques*
- **Expected files:**
  - `data/raw/train.csv` (must include `SalePrice`)
  - `data/raw/test.csv` (must not include `SalePrice`)

## Technologies
- Python, pandas, numpy
- scikit-learn (Pipeline, ColumnTransformer, CV)
- CatBoostRegressor
- matplotlib, seaborn, jupyter
- pytest, joblib

## Project Structure
```text
House_Price_Prediction/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── reports/
│   └── figures/
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── tests/
├── .gitignore
├── requirements.txt
└── README.md
```

## ML Workflow
1. Load and validate datasets (`src/data_loader.py`).
2. Engineer domain features (`src/feature_engineering.py`):
   - `TotalSF`: `TotalBsmtSF + 1stFlrSF + 2ndFlrSF`
   - `HouseAge`: `YrSold - YearBuilt`
   - `RemodAge`: `YrSold - YearRemodAdd`
   - `GarageAge`: `YrSold - GarageYrBlt` (fallback to `YearBuilt` when missing)
   - `TotalBathrooms`: full/half above-ground + basement bathrooms
3. Build preprocessing via sklearn `ColumnTransformer` (`src/preprocessing.py`).
4. Train CatBoost with optional randomized search (`src/train.py`) using negative RMSE scoring.
5. Evaluate regression metrics and run feature ablation (`src/evaluate.py`).
6. Predict on new data and reverse log target transform with `np.expm1` when enabled (`src/predict.py`).

## Evaluation Metrics
- **RMSE:** main error metric for large deviations.
- **MAE:** average absolute error.
- **R²:** explained variance.
- **RMSLE:** log-space error (when values are non-negative).

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows PowerShell
pip install -r requirements.txt
```

## How to Run
### 1) Run tests
```bash
pytest -q
```

### 2) Train model (from Python)
```python
from src.train import TrainingConfig, run_training

config = TrainingConfig(
    train_path="data/raw/train.csv",
    test_path="data/raw/test.csv",
    model_output_path="models/catboost_pipeline.joblib",
    tune_hyperparameters=True,
)
model_bundle = run_training(config)
print(model_bundle["best_cv_rmse"])
```

### 3) Predict (from Python)
```python
from src.predict import predict_from_csv

pred_df = predict_from_csv(
    model_path="models/catboost_pipeline.joblib",
    input_csv_path="data/raw/test.csv",
    output_csv_path="data/processed/predictions.csv",
)
```

## EDA Notes
The repository keeps notebook space under `notebooks/` and figure output under `reports/figures/`. Use these to store concise EDA focused on distributions, missingness, and feature-target relationships.

## Results and Limitations
- No fixed metrics are claimed in this README; performance depends on the dataset split and tuning configuration.
- Reproducibility requires local availability of Kaggle train/test files.

## Future Improvements
- Add CLI wrappers for train/evaluate/predict modules.
- Add richer experiment tracking.
- Add model comparison baselines.

## Author
Hyder (GitHub: [@coderhyder17](https://github.com/coderhyder17))
