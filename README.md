# Loan Prediction — Practice Problem (Python)

> A clean, beginner-friendly machine learning pipeline that predicts whether a loan application will be approved, using an XGBoost classifier tuned with GridSearchCV.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-337AB7)

## Overview

This project tackles the classic Loan Prediction practice problem: given an applicant's demographic and financial information, predict the loan status (`Loan_Status`). It is a compact, well-structured example of a production-style scikit-learn workflow — EDA, preprocessing, a single reusable `Pipeline`, hyperparameter tuning, and evaluation — all in one readable script (`main.py`).

It is a good starting point for beginners and intermediates who want to see how the pieces of a real ML project fit together.

## Key Features

- **Exploratory Data Analysis** — target distribution plots and a correlation heatmap (Seaborn / Matplotlib)
- **Data cleaning & preprocessing** — categorical features one-hot encoded, numeric features passed through, via a `ColumnTransformer`
- **Single ML pipeline** — preprocessing + model in one scikit-learn `Pipeline`, avoiding train/test leakage
- **Hyperparameter tuning** — `GridSearchCV` (3-fold CV, accuracy) over XGBoost `n_estimators`, `max_depth`, `learning_rate` and `subsample`
- **Proper evaluation** — stratified 80/20 train/test split, test accuracy, classification report and confusion matrix

## Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, XGBoost
- **Dataset:** `loan_train.csv` — the standard loan-prediction training file (place it next to `main.py` before running)

## Project Structure

```text
Loan-prediction-practice-problem-using-python/
├── main.py      # EDA → preprocessing → XGBoost pipeline → tuning → evaluation
└── README.md
```

## Getting Started

```bash
git clone https://github.com/datascientist970/Loan-prediction-practice-problem-using-python.git
cd Loan-prediction-practice-problem-using-python
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

Download the loan-prediction training dataset and save it as `loan_train.csv` in the project folder.

## Usage

```bash
python main.py
```

The script prints the best hyperparameters and cross-validation score found by GridSearchCV, followed by the test-set accuracy, classification report and a confusion-matrix heatmap.

## Possible Future Improvements

- SHAP-based explainability and feature-importance visualisation
- Deployment as a small Flask / FastAPI service
- Hyperparameter search with Optuna

## Author

**Muhammad Dawood** — AI & Data Science Engineer
GitHub: https://github.com/datascientist970 · Mardan, Pakistan
LinkedIn: https://www.linkedin.com/in/dawood406/
