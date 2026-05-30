# Fraud Detection System — Government Agency
> **Course:** DLBDSMTP01 | **Type:** MLOps Project | **University:** IU International University of Applied Sciences

---

## Table of Contents
1. [Project Scope](#project-scope)
2. [System Architecture](#system-architecture)
3. [Project Structure](#project-structure)
4. [Dataset](#dataset)
5. [Machine Learning Models](#machine-learning-models)
6. [REST API](#rest-api)
7. [Monitoring with MLflow](#monitoring-with-mlflow)
8. [MLOps Pipeline — GitHub Actions](#mlops-pipeline--github-actions)
9. [Data Drift Detection](#data-drift-detection)
10. [Security](#security)
11. [Getting Started](#getting-started)
12. [API Usage](#api-usage)

---

## Project Scope

A government financial aid agency processes thousands of applications per month through an online system. In recent years, the number of fraudulent applications has increased significantly, preventing legitimate applicants from receiving support.

This project designs and implements an **automated fraud detection system** that:
- Automatically detects fraud in incoming applications with a probability score
- Integrates seamlessly with the agency's existing systems via a RESTful API
- Retrains automatically when new data arrives or data drift is detected
- Tracks all experiments and model performance with MLflow
- Is secured with API key authentication

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│                                                                 │
│   fraud_dataset.csv  ──►  generate_data.py  ──►  month_XX.csv  │
│   (base dataset)          (SDV synthesizer        (12 monthly   │
│                            + drift injection)      data files)  │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING LAYER                             │
│                                                                 │
│   train.py  ──►  preprocessing_data()  ──►  train_models()     │
│                                              │                  │
│                                              ▼                  │
│                                         MLflow Tracking         │
│                                         (metrics, params,       │
│                                          artifacts)             │
│                                              │                  │
│                                              ▼                  │
│                                         app/model/*.pkl         │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       MLOPS LAYER                               │
│                                                                 │
│   GitHub Actions  ──►  retrain.yml                              │
│   (monthly cron)        │                                       │
│                         ├──► generate_data.py                   │
│                         ├──► detect_drift()                     │
│                         ├──► train.py --data month_XX.csv       │
│                         └──► commit new .pkl files to repo      │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SERVING LAYER                              │
│                                                                 │
│   FastAPI (main.py)                                             │
│   ├── GET  /                    (health check — public)         │
│   ├── POST /predict             (batch CSV predictions)         │
│   ├── POST /predict-an-instance (single transaction)            │
│   └── POST /reload              (reload models from disk)       │
│                                                                 │
│   Authentication: X-API-Key header (all endpoints except /)     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
fraud-detection-mlops/
│
├── app/
│   ├── model/
│   │   ├── model.py                  # Core ML functions
│   │   ├── logistic_regression.pkl   # Trained model
│   │   ├── random_forest.pkl         # Trained model
│   │   └── gradient_boosting_classifier.pkl
│   ├── data/
│   │   └── fraud_dataset.csv         # Base dataset
│   └── logs/
│       └── app.log                   # API request logs
│
├── data/
│   ├── fraud_dataset.csv             # Original dataset
│   ├── fraud_synthetic.csv           # SDV synthetic data
│   ├── data_params.json              # SDV metadata
│   └── monthly/
│       ├── month_01.csv              # Simulated monthly data
│       ├── month_02.csv
│       └── ... (up to month_12.csv)
│
├── .github/
│   └── workflows/
│       └── retrain.yml               # GitHub Actions pipeline
│
├── main.py                           # FastAPI application
├── train.py                          # Standalone training script
├── generate_data.py                  # Synthetic data + drift generation
├── requirements.txt                  # Python dependencies
├── .env                              # API key (never committed)
├── .gitignore
└── README.md
```

---

## Dataset

The project uses **synthetic financial transaction data** generated with [SDV (Synthetic Data Vault)](https://sdv.dev/) based on a realistic fraud dataset. Each record represents one financial aid application with the following features:

| Feature | Type | Description |
|---------|------|-------------|
| `transaction_id` | int | Unique transaction identifier |
| `user_id` | int | Applicant identifier |
| `amount` | float | Transaction amount |
| `hour` | int | Hour of day (0–23) |
| `device_risk_score` | float | Risk score of the device (0–1) |
| `ip_risk_score` | float | Risk score of the IP address (0–1) |
| `transaction_type` | str | Online, QR, ATM, POS |
| `merchant_category` | str | Clothing, Electronics, Food, Grocery, Travel |
| `country` | str | UK, US, DE, FR, NG, TR |
| `is_fraud` | int | Target variable (1 = Fraud, 0 = Legitimate) |

**Dataset size:** 10,000 rows | **Fraud rate:** ~5%

---

## Machine Learning Models

Three classifiers are trained and compared on every training run:

| Model | Key Parameters |
|-------|---------------|
| Logistic Regression | `solver=lbfgs`, `max_iter=1000` |
| Random Forest | `n_estimators=100`, `random_state=42` |
| Gradient Boosting | `n_estimators=100`, `learning_rate=1.0`, `max_depth=1` |

**Prediction strategy:** Ensemble averaging — the final fraud probability is the mean confidence score across all three models. A transaction is flagged as fraud if the average score exceeds **0.6**.

**Evaluation metrics tracked in MLflow:**
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC
- Confusion Matrix

---

## REST API

The API is built with **FastAPI** and served with **Uvicorn**.

### Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/` | No | Health check and project info |
| POST | `/predict` | Yes | Batch prediction from CSV upload |
| POST | `/predict-an-instance` | Yes | Single transaction prediction |
| POST | `/reload` | Yes | Reload models from disk without restart |

### Request & Response

**`POST /predict`** — Upload a CSV file with transaction data:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "X-API-Key: your-api-key" \
  -F "file=@transactions.csv"
```
```json
{
  "is_fraud": [0, 1, 0, 0, 1],
  "confidence_score": [0.12, 0.87, 0.05, 0.23, 0.91]
}
```

**`POST /predict-an-instance`** — Single transaction JSON body:
```bash
curl -X POST "http://127.0.0.1:8000/predict-an-instance" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "Transaction_id": 1001,
    "User_id": 42,
    "Amount": 1500.00,
    "Hour": 23,
    "Device_risk_score": 0.85,
    "Ip_risk_score": 0.76,
    "Transaction_type": "Online",
    "Merchant_category": "Electronics",
    "Country": "NG"
  }'
```
```json
{
  "is_fraud": 1,
  "confidence_score": 0.89
}
```

---

## Monitoring with MLflow

MLflow tracks every training run automatically:

- **Experiment:** `Fraud Detection - Training`
- **Tracked per run:** model parameters, accuracy, F1, AUC, confusion matrix
- **Production metrics:** fraud rate, batch size, mean amount per prediction batch

**Start the MLflow UI:**
```bash
mlflow ui
```
Then open: `http://127.0.0.1:5000`

---

## MLOps Pipeline — GitHub Actions

The retraining pipeline is defined in `.github/workflows/retrain.yml` and runs automatically on the **1st of every month at midnight UTC**.

### Pipeline Steps

```
1. Checkout repository
2. Set up Python 3.10
3. Install dependencies
4. Determine month number (scheduled or manual input)
5. Run generate_data.py → produce monthly CSV + detect drift
6. Run train.py --data data/monthly/month_XX.csv
7. Commit updated .pkl model files back to repository
```

### Manual Trigger

You can also trigger the pipeline manually from the GitHub Actions tab with a specific month number (1–12) for testing.

### GitHub Secrets Required

| Secret | Value |
|--------|-------|
| `GITHUB_TOKEN` | Auto-provided by GitHub |
| `MLFLOW_TRACKING_URI` | `mlruns` (file-based for CI) |

---

## Data Drift Detection

Drift is detected using the **Kolmogorov-Smirnov (KS) test** on four numerical features:

| Feature | Drift Strength per Month |
|---------|--------------------------|
| `amount` | 0.03 |
| `hour` | 0.003 |
| `device_risk_score` | 0.008 |
| `ip_risk_score` | 0.008 |

If the KS test p-value falls below **0.05** for any feature, drift is flagged and retraining is triggered. Results are printed during the pipeline run:

```
amount           KS=0.0312  p=0.0031  🔴 DRIFT
hour             KS=0.0089  p=0.4821  🟢 OK
device_risk_score KS=0.0201  p=0.0412  🔴 DRIFT
ip_risk_score    KS=0.0156  p=0.1203  🟢 OK
```

---

## Security

- All prediction endpoints are protected with an **API key** passed in the `X-API-Key` request header
- The API key is stored in a `.env` file locally and never committed to the repository
- Unauthorized requests receive a `403 Forbidden` response
- All requests and errors are logged to `app/logs/app.log`

---

## Getting Started

### Prerequisites
- Python 3.10+
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/fatimagulomova/fraud-detection-mlops.git
cd fraud-detection-mlops

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file with your API key
echo "API_KEY=your-secret-key-here" > .env

# 4. Generate synthetic data and monthly files
python generate_data.py

# 5. Start MLflow tracking server
mlflow ui

# 6. Train the models (in a new terminal)
python train.py

# 7. Start the API (in a new terminal)
uvicorn main:app --reload
```

### Access Points

| Service | URL |
|---------|-----|
| FastAPI | `http://127.0.0.1:8000` |
| Swagger UI | `http://127.0.0.1:8000/docs` |
| MLflow UI | `http://127.0.0.1:5000` |

---

## API Usage

Generate a secure API key:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Add it to `.env`:
```
API_KEY=the-generated-key
```

Pass it in every request header:
```
X-API-Key: the-generated-key
```

Or use the 🔒 **Authorize** button in Swagger UI at `http://127.0.0.1:8000/docs`.