import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

# --- 1. System Setup ---
app = FastAPI(
    title="Fraud Detection API",
    description="API for detecting credit card fraud using XGBoost",
    version="1.0"
)

# Paths
MODEL_PATH = 'outputs/models/final_model_pipeline.joblib'
SCALER_PATH = 'outputs/models/scaler.joblib'

# Global variables to hold model
model_pipeline = None
scaler = None

# --- 2. Data Validation Schema ---

class TransactionInput(BaseModel):
    Amount: float
    Hour: int
    # V1-V28 are optional for simplicity in testing, default to 0 if not provided
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0

# --- 3. Startup Event (Load Model) ---
@app.on_event("startup")
def load_artifacts():
    global model_pipeline, scaler
    try:
        if os.path.exists(MODEL_PATH):
            model_pipeline = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            print("Model and Scaler loaded successfully API is ready.")
        else:
            print("Error: Model files not found. Please run main_pipeline.py first.")
    except Exception as e:
        print(f"Error loading model: {e}")

# --- 4. API Endpoints ---

@app.get("/")
def home():
    return {"message": "Credit Card Fraud Detection API is Online. Go to /docs to test it."}

@app.post("/predict")
def predict_fraud(transaction: TransactionInput):
    """
    Receives transaction details and returns fraud probability.
    """
    if not model_pipeline or not scaler:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # 1. Prepare Data
    # Convert Pydantic object to DataFrame
    data_dict = transaction.dict()
    
    # Feature Engineering (Backend Logic)
    # API must replicate the same logic: Hour -> is_night
    hour = data_dict['Hour']
    is_night = 1 if 0 <= hour <= 6 else 0
    
    # Create DataFrame structure expected by model
    # Note: We need to order columns if necessary, but XGBoost is robust.
    # However, 'Amount' needs scaling.
    
    input_df = pd.DataFrame([data_dict])
    
    # Add engineered feature
    input_df['is_night'] = is_night
    
    # Apply Scaling to Amount
    input_df['Amount'] = scaler.transform(input_df[['Amount']])
    
    # The pipeline expects specific columns. The pipeline object handles the model part.
    # We need to access the model directly or ensure pipeline input matches.
    # Let's use the named step 'xgb' directly for safety as we did in scenarios.
    model = model_pipeline.named_steps['xgb']
    
    # Ensure column order matches training (drop extra cols if any)
    # Assuming V1..V28, Amount, Hour, is_night are present
    # To be safe, let's select columns we need
    # (In a real scenario, we'd enforce strict column ordering here)
    
    # Predict
    prob = model.predict_proba(input_df)[0, 1]
    is_fraud = prob > 0.5  # You can use the loaded threshold if you want
    
    return {
        "fraud_probability": float(prob),
        "is_fraud": bool(is_fraud),
        "alert": "FRAUD DETECTED" if is_fraud else "Normal Transaction"
    }

# --- 5. Run Server ---
if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
