import os
import sys
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.model.model import (
    preprocessing_data,
    train_models_with_mlflow,
    classification_report_vis,
    feature_importance_vis
)

# ==================== ARGUMENT PARSING ====================

parser = argparse.ArgumentParser(description="Train fraud detection models")
parser.add_argument(
    "--data",
    type=str,
    # FIX: default points to base dataset if no monthly file is specified
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "fraud_dataset.csv"),
    help="Path to the training CSV file"
)
args = parser.parse_args()

# ==================== LOAD DATA ====================

# FIX: resolve path relative to script location, not working directory
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.data) if not os.path.isabs(args.data) else args.data

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found: {data_path}")

print(f"Loading data from: {data_path}")
df = pd.read_csv(data_path)
print(f"Data loaded: {len(df)} rows")

# ==================== PREPROCESSING ====================

X_train, X_test, y_train, y_test = preprocessing_data(df=df)

# ==================== MODELS ====================

models = [
    (
        "Logistic Regression",
        {
            "class_weight": None,
            "random_state": 42,
            "solver": "lbfgs",
            "max_iter": 100
        },
        LogisticRegression(),
    ),
    (
        "Random Forest",
        {
            "n_estimators": 100,
            "random_state": 42
        },
        RandomForestClassifier(),
    ),
    (
        "Gradient Boosting Classifier",
        {
            "n_estimators": 100,
            "learning_rate": 1.0,
            "max_depth": 1,
            "random_state": 42
        },
        GradientBoostingClassifier(),
    ),
]

# ==================== TRAINING ====================

print("Starting training...")
y_preds, y_probas = train_models_with_mlflow(
    models=models,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test
)
print("Training complete.")

# ==================== EVALUATION ====================

for y_pred, y_proba in zip(y_preds, y_probas):
    classification_report_vis(y_test=y_test, y_pred=y_pred)

for name, params, model in models:
    if hasattr(model, 'feature_importances_'):
        feature_importance_vis(model=model, X=X_train)
    else:
        print(f"{name} does not support feature_importances_, skipping.")

print("All done.")