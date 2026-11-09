# House Price Prediction

A machine learning project that predicts residential house sale prices
using the Ames Housing dataset. The project covers exploratory data
analysis, data preprocessing, feature engineering, model development,
hyperparameter tuning, and prediction generation.

## Table of Contents

-   [Project Overview](#project-overview)
-   [Objectives](#objectives)
-   [Dataset](#dataset)
-   [Project Workflow](#project-workflow)
-   [Feature Engineering](#feature-engineering)
-   [Data Preprocessing](#data-preprocessing)
-   [Model Development](#model-development)
-   [Feature Ablation](#feature-ablation)
-   [Prediction and Submission](#prediction-and-submission)
-   [Technologies and Libraries](#technologies-and-libraries)
-   [Getting Started](#getting-started)
-   [Reproducibility Notes](#reproducibility-notes)

## Project Overview

The goal of this project is to build a regression model that estimates
the sale price of residential properties from their characteristics.

The work is organized as a Jupyter Notebook and follows a typical
end-to-end machine learning workflow:

1.  Understand the dataset and prediction target.
2.  Explore distributions, relationships, and data quality.
3.  Handle missing values and prepare numerical and categorical
    features.
4.  Create additional features that may help explain property value.
5.  Train and tune a regression model.
6.  Evaluate model performance.
7.  Generate predictions in the format required for the competition.

## Objectives

-   Explore the Ames Housing data and understand the variables related
    to sale price.
-   Identify missing values and prepare the data for machine learning.
-   Apply appropriate preprocessing to numerical and categorical
    columns.
-   Engineer meaningful property-related features.
-   Train and tune a CatBoost regression model.
-   Use cross-validation to compare feature-set variations.
-   Generate predictions for the competition test data.

## Dataset

This project uses the **Ames Housing** dataset from Kaggle's House
Prices: Advanced Regression Techniques competition.

-   Competition: [House Prices: Advanced Regression
    Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
-   Training data: Includes property features and the target column,
    `SalePrice`.
-   Test data: Includes property features without the target column.
-   Identifier: `Id`, used to associate predictions with the
    corresponding test records.
-   Target: `SalePrice`, the property's sale price.

The target is log-transformed during model training to work with the
skewed sale-price distribution. Predictions are converted back to the
original price scale using `numpy.expm1`.

> **Dataset access:** Download the competition files from Kaggle and
> place them in the data directory expected by the notebook. Follow the
> file paths used in the notebook if your local folder structure is
> different.

## Project Workflow

### 1. Data Understanding and Exploratory Data Analysis

The initial analysis focuses on understanding the dataset before
modeling.

Typical investigation areas include:

-   Dataset dimensions and column data types.
-   Numerical and categorical variables.
-   Missing-value counts and patterns.
-   Target-variable distribution.
-   Relationships between selected features and `SalePrice`.
-   Correlations among numerical variables.
-   Potentially unusual observations that may need further
    investigation.

### 2. Data Cleaning and Missing-Value Handling

Missing values are reviewed and handled as part of the preprocessing
workflow.

The treatment of a missing value should depend on what it represents:

-   Some missing values indicate that a property does not have a
    particular feature.
-   Other missing values represent information that was not recorded.
-   Numerical and categorical columns may require different imputation
    strategies.

Preprocessing is organized so that transformations can be applied
consistently during model training and prediction.

### 3. Feature Engineering

The project creates additional features that summarize useful property
characteristics. These engineered features are:

  Feature            Description
  ------------------ ---------------------------------------
  `TotalSF`          Combined square footage measure
  `HouseAge`         Age of the house at the time of sale
  `RemodAge`         Time since the most recent remodeling
  `GarageAge`        Age of the garage
  `TotalBathrooms`   Combined bathroom count

These features are intended to represent property size, age, renovation
history, and facilities in a more direct way.

### 4. Data Preprocessing

A preprocessing pipeline is used to prepare features for modeling.

The workflow includes:

-   Handling missing values.
-   Separating numerical and categorical columns.
-   Applying suitable transformations to each feature type.
-   Combining the transformations with a `ColumnTransformer`.
-   Integrating preprocessing and the estimator through a scikit-learn
    `Pipeline`.

Using pipelines helps keep preprocessing consistent between training and
prediction and reduces the risk of data leakage when transformations are
fitted within cross-validation.

### 5. Model Training and Hyperparameter Tuning

The project uses **CatBoostRegressor** for regression.

The model is tuned using a parameter search workflow. The selected
parameters are taken from the best-performing search configuration and
used to build the tuned model.

The training target is the log-transformed sale price:

``` python
y_train_log = np.log1p(y_train)
```

This transformation reduces the effect of the highly skewed target
distribution. Predictions are converted back to the original price scale
with:

``` python
predictions = np.expm1(predictions_log)
```

The notebook contains the implementation details and parameter values
used in the experiment.

### 6. Model Evaluation

Model performance is assessed using regression evaluation and
cross-validation.

The project uses RMSE-based evaluation during model comparison and
feature-ablation experiments.

-   **RMSE (Root Mean Squared Error):** Measures the typical prediction
    error while penalizing larger errors more strongly.
-   **Cross-validation:** Evaluates model performance across multiple
    validation folds rather than relying on a single split.

The final numerical scores should be taken from the latest executed
notebook outputs. They are intentionally not stated here because a
verified final score was not included in the project details used to
prepare this README.

## Feature Ablation

Feature ablation is used to investigate whether each engineered feature
contributes to model performance.

The experiment:

1.  Trains/evaluates a model using the full feature set.
2.  Removes one engineered feature at a time.
3.  Evaluates each reduced feature set using 3-fold cross-validation.
4.  Compares the mean negative RMSE scores.

The engineered features examined individually are:

-   `TotalSF`
-   `HouseAge`
-   `RemodAge`
-   `GarageAge`
-   `TotalBathrooms`

This comparison helps assess the contribution of each engineered
feature. A feature should not be considered useful solely because it
sounds meaningful; its effect should be judged using validation results.

## Prediction and Submission

After model training and evaluation:

1.  Generate predictions for the competition test data.
2.  Reverse the log transformation using `numpy.expm1`.
3.  Pair predictions with the corresponding test-set `Id` values.
4.  Save the output as `submission.csv`.

The submission file uses the competition's required columns:

  Column        Description
  ------------- --------------------------------------------------
  `Id`          Test property's identifier
  `SalePrice`   Predicted sale price on the original price scale

Example structure:

``` csv
Id,SalePrice
1461, ...
1462, ...
1463, ...
```

The values above are illustrative placeholders, not actual model
predictions.

## Technologies and Libraries

The project uses the Python data science ecosystem.

-   **Python** --- Programming language.
-   **Jupyter Notebook** --- Interactive development and
    experimentation.
-   **Pandas** --- Data loading and manipulation.
-   **NumPy** --- Numerical operations and target transformation.
-   **Matplotlib** --- Data visualization.
-   **Seaborn** --- Statistical visualization.
-   **Scikit-learn** --- Preprocessing, pipelines, cross-validation, and
    model-selection utilities.
-   **CatBoost** --- Gradient-boosting regression model.

## Getting Started

### 1. Clone the repository

``` bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

Replace the placeholders with your GitHub repository URL and folder
name.

### 2. Create and activate a virtual environment (recommended)

**Windows:**

``` bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the required libraries

``` bash
python -m pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scikit-learn catboost jupyter
```

If the notebook imports additional libraries, install those dependencies
as well.

### 4. Download the dataset

-   Open the [Kaggle competition
    page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).
-   Download the competition data.
-   Extract the files.
-   Place the training and test CSV files in the locations referenced by
    the notebook.

You may need to join the competition and accept its rules before
downloading the data.

### 5. Run the notebook

Start Jupyter:

``` bash
jupyter notebook
```

Open `house-price-prediction.ipynb` and run the cells in order.

## Reproducibility Notes

-   Run notebook cells from top to bottom to ensure variables and
    transformations are created in the correct order.
-   Confirm that the dataset paths match your local directory structure.
-   Use the same preprocessing and feature-engineering steps for
    training and test data.
-   Fit preprocessing steps using training data only; do not use target
    information from validation or test data.
-   Record the final validation results from the executed notebook when
    updating this README.
-   Do not commit private credentials, local environment folders, or
    unrelated personal data to the repository.

------------------------------------------------------------------------

## Author

**Adnan Hyder**

GitHub: [@coderhyder17](https://github.com/coderhyder17)

------------------------------------------------------------------------

*This README describes the workflow and techniques documented for the
project. Refer to the Jupyter Notebook for the complete implementation,
experiment configurations, and executed results.*
