import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import mlflow

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

__version__ = "0.1.0"

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000/"))
mlflow.set_experiment('Fraud Detection - Training')

# =================================================================== DATA PREPROCESSING ===================================================================

def preprocessing_data(df):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    df_encoded = pd.get_dummies(df, columns=cat_cols)

    y = df['is_fraud']
    # FIX: 'month' added to drop list — it was only dropped in make_prediction before
    X = df_encoded.drop(columns=[c for c in ['is_fraud', 'transaction_id', 'user_id', 'month'] if c in df_encoded.columns])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


# ========================================================================= MODEL TRAINING ==================================================================

def train_models_with_mlflow(models, X_train, X_test, y_train, y_test):
    y_preds, y_probas = [], []

    for i, element in enumerate(models):
        model_name = element[0]
        params = element[1]
        model = element[2]

        model.set_params(**params)

        with mlflow.start_run(run_name=model_name):
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            y_preds.append(y_pred)
            y_probas.append(y_proba)

            eval_dataset = pd.DataFrame({
                'prediction': y_pred,
                'prediction_proba': y_proba,
                'target': y_test
            })

            mlflow.models.evaluate(
                data=eval_dataset,
                predictions="prediction",
                targets="target",
                model_type="classifier",
            )

            mlflow.log_params(params)
            mlflow.sklearn.log_model(model, model_name)

            safe_name = model_name.lower().replace(" ", "_")
            with open(f'app/model/{safe_name}.pkl', 'wb') as file:
                pickle.dump(model, file)

    return y_preds, y_probas


# ======================================================================= EVALUATION VISUALIZATIONS ======================================================

def classification_report_vis(y_test, y_pred):
    
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def feature_importance_vis(model, X):
    
    if not hasattr(model, 'feature_importances_'):
        print(f"{type(model).__name__} does not support feature_importances_, skipping.")
        return

    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    plt.figure(figsize=(20, 9))
    # FIX: use y='feature' instead of hue='feature' for proper bar chart
    sns.barplot(x='importance', y='feature', hue='feature', data=feat_df, palette='viridis', legend=False)
    plt.title("Feature Importance")
    plt.show()


def visualize_predictions(data):
    # FIX: guard for missing 'amount' column (dropped during encoding)
    if 'amount' not in data.columns:
        print("Column 'amount' not available for visualization.")
        return

    plt.figure(figsize=(10, 5))
    # FIX: cast is_fraud to str so seaborn treats it as categorical
    sns.barplot(x=data.index, y=data['amount'], hue=data['is_fraud'].astype(str), dodge=False)
    plt.title("Predicted Fraud vs Legit Transactions")
    plt.xlabel("Transaction Index")
    plt.ylabel("Amount")
    plt.show()


# ================================================================= PREDICTION ================================================================

def make_prediction(models, data: pd.DataFrame):
    data = data.copy()

    # FIX: drop columns once before the loop (not inside it)
    cols_to_drop = [c for c in ['transaction_id', 'user_id', 'month', 'is_fraud'] if c in data.columns]
    data = data.drop(columns=cols_to_drop)

    cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    data_encoded = pd.get_dummies(data, columns=cat_cols)

    # FIX: accumulate scores separately so reindex doesn't overwrite predictions
    all_scores = []

    for model in models:
        model_input = data_encoded.reindex(columns=model.feature_names_in_, fill_value=0)
        confidence_score = model.predict_proba(model_input)[:, 1]
        all_scores.append(confidence_score)

    # Average confidence across all models (ensemble)
    avg_score = np.mean(all_scores, axis=0)

    result = pd.DataFrame({
        'is_fraud': (avg_score >= 0.6).astype(int),
        'fraud_probability': avg_score
    })

    return result
