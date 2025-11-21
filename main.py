# ----------------------------
# Loading Header files
# ----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier

# ----------------------------
# Loading dataset
# ----------------------------

df = pd.read_csv("loan_train.csv")
df.head()

# ----------------------------
# Basic EDA
# ----------------------------
df.isnull().sum()
sns.countplot(data=df, x="Loan_Status")
plt.show()
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.show()

# ----------------------------
# Preprocessing
# ----------------------------
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cat_features = X.select_dtypes(include=["object"]).columns.tolist()
num_features = X.select_dtypes(exclude=["object"]).columns.tolist()

# ----------------------------
# Pipeline
# ----------------------------
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("num", "passthrough", num_features)
    ]
)


xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42
)

pipeline = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", xgb)
])

# ----------------------------
# hyperameter Tuning
# ----------------------------

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.01, 0.1],
    "model__subsample": [0.8, 1],
}

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=2
)

# ----------------------------
# Training
# ----------------------------
grid.fit(X_train, y_train)

# ----------------------------
#Best model and parameters
# ----------------------------
print("Best parameters:", grid.best_params_)
print("Best CV Accuracy:", grid.best_score_)

# ----------------------------
# Evaluation
# ----------------------------
y_pred = grid.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
plt.show()

