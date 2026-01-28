from fastapi import FastAPI
from api.schemas import CustomerData, PredictionResponse
from api.predict import predict_churn

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using a trained ML model",
    version="1.0"
)

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    return predict_churn(data.dict())