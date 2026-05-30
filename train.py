# train.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from model.model import preprocessing_data, train_models_with_mlflow, classification_report_vis, feature_importance_vis

df = pd.read_csv("app/data/fraud_dataset.csv")
X_train, X_test, y_train, y_test = preprocessing_data(df=df)

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

y_preds, y_probas = train_models_with_mlflow(models=models, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

for y_pred, y_proba in zip(y_preds, y_probas):
    classification_report_vis(y_test=y_test, y_pred=y_pred)

for name, params, model in models:
    if hasattr(model, 'feature_importances_'):
        feature_importance_vis(model=model, X=X_train)
    else:
        print(f"{name} does not support feature_importances_, skipping.")