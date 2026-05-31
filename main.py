from fastapi import FastAPI, File, UploadFile, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
import pandas as pd
import os, io, sys, mlflow
import pickle, glob, logging
from dotenv import load_dotenv


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.model.model import make_prediction

# Logging
logging.basicConfig(
    filename="app/logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

load_dotenv()  # loads .env file

API_KEY = os.getenv("API_KEY")  # read key from environment
API_KEY_NAME = "X-API-Key"      # the header name clients must send
 
if not API_KEY:
    raise RuntimeError("API_KEY not set. Add it to your .env file.")
 
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
 
async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        logging.warning("Unauthorized access attempt.")
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API key. Pass it as 'X-API-Key' in your request headers."
        )
    return api_key

mlflow.set_tracking_uri('http://127.0.0.1:5000/')
mlflow.set_experiment('Fraud Detection - Production')


# ==================================================================================== PYDANTIC MODELS ===============================================================================

class Input(BaseModel):
    Transaction_id: int = Field(..., description="The ID of your transaction")
    User_id: int = Field(..., description="Your ID")
    Amount: float = Field(..., description="The amount of money you want to transfer")
    Hour: int = Field(..., description="The current time")
    Device_risk_score: float = Field(..., description="Your Device risk score")
    Ip_risk_score: float = Field(..., description="Your IP risk score")
    Transaction_type: str = Field(..., description="Choose the Transaction type: Online, QR, ATM, POS")
    Merchant_category: str = Field(..., description="On what you spend money: Clothing, Electronics, Food, Grocery, Travel")
    Country: str = Field(..., description="Your country: UK, US, DE, FR, NG, TR")


class PredictionOutput(BaseModel):
    is_fraud: int = Field(..., description="1 is Fraud, 0 is Legitimate")
    confidence_score: float = Field(..., description="The model's confidence")


class PredictionOut(BaseModel):
    is_fraud: list[int]
    confidence_score: list[float]


# ============================================================================================== LOAD MODELS ==========================================================================
 
# Load only your three specific models
MODEL_NAMES = [
    "app/model/logistic_regression.pkl",
    "app/model/random_forest.pkl",
    "app/model/gradient_boosting_classifier.pkl"
]

model_list = []
for file_path in MODEL_NAMES:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            model_list.append(pickle.load(f))
    else:
        logging.warning(f"Model file not found: {file_path}")

# ================================================================================================= APP ===============================================================================

app = FastAPI()


@app.get("/")
async def root():
    return {
        "Name": "Fraud Detection In A Government Agency",
        "description": (
            "This project designs and implements an automated fraud detection "
            "system for a government financial aid agency. "
            "The system uses multiple classifier models trained on synthetic financial "
            "transaction data to predict the probability of fraud in incoming "
            "applications. The model is served as a RESTful API built with "
            "FastAPI and monitored using MLflow for experiment tracking and model versioning."
        ),
    }


@app.post("/predict", response_model=PredictionOut, dependencies=[Depends(verify_api_key)])
async def predict(file: UploadFile = File(...)):
    try:
        logging.info(f"Received file: {file.filename}")

        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")

        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Validate expected columns before predicting
        expected_cols = [
            'transaction_id', 'user_id', 'amount', 'hour',
            'device_risk_score', 'ip_risk_score',
            'transaction_type', 'merchant_category', 'country'
        ]

        missing = [c for c in expected_cols if c not in df.columns]

        if missing:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

       
        result = make_prediction(models=model_list, data=df)

        try:
            with mlflow.start_run(run_name="prediction_batch"):
                mlflow.log_metric("fraud_rate", float(result['is_fraud'].mean()))
                mlflow.log_metric("batch_size", int(len(result)))
                # FIX: log amount from original df before encoding drops it
                mlflow.log_metric("mean_amount", float(df['amount'].mean()))

        except Exception as mlflow_err:
            logging.warning(f"MLflow logging failed: {mlflow_err}")

        
        return {
            "is_fraud": result['is_fraud'].astype(int).tolist(),
            "confidence_score": result['fraud_probability'].astype(float).tolist()
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-an-instance", response_model=PredictionOutput, dependencies=[Depends(verify_api_key)])
async def predict_instance(features: Input):
    try:
        df = pd.DataFrame([{
            "transaction_id": int(features.Transaction_id),
            "user_id": int(features.User_id),
            "amount": float(features.Amount),
            "hour": int(features.Hour),
            "device_risk_score": float(features.Device_risk_score),
            "ip_risk_score": float(features.Ip_risk_score),
            "transaction_type": str(features.Transaction_type),
            "merchant_category": str(features.Merchant_category),
            "country": str(features.Country)
        }], index=[0])

        result = make_prediction(models=model_list, data=df)

        return {
            "is_fraud": int(result['is_fraud'].iloc[0]),
            "confidence_score": float(result['fraud_probability'].iloc[0])
        }

    except Exception as e:
        logging.exception("Instance prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload", dependencies=[Depends(verify_api_key)])
async def reload_models():
    global model_list
    try:
        new_models = []
        for file_path in glob.glob("app/model/*.pkl"):
            with open(file_path, "rb") as f:
                new_models.append(pickle.load(f))

        if not new_models:
            raise HTTPException(status_code=404, detail="No model files found in app/model/")

        model_list = new_models
        logging.info(f"Models reloaded successfully. {len(model_list)} models loaded.")

        return {
            "status": "success",
            "models_loaded": len(model_list),
            "message": "Models reloaded from disk successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Model reload failed")
        raise HTTPException(status_code=500, detail=str(e))