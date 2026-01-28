import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "Models" / "final_churn_model.pkl"

model = joblib.load(MODEL_PATH)

THRESHOLD = 0.4  # business decision threshold

def predict_churn(data: dict):
    df = pd.DataFrame([data])
    proba = model.predict_proba(df)[0, 1]
    proba = round(float(proba),2)
    prediction = int(proba >= THRESHOLD)

    return {
        "churn_probability": float(proba),
        "churn_prediction": prediction
    }