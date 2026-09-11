
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting telecom customer churn",
    version="1.0.0"
)

# Load trained model
model = joblib.load("/content/customer_churn_model.pkl")


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }


@app.post("/predict")
def predict_churn(customer: CustomerData):

    customer_dict = customer.model_dump()
    customer_df = pd.DataFrame([customer_dict])

    prediction = model.predict(customer_df)[0]
    probability = model.predict_proba(customer_df)[0][1]

    if probability < 0.30:
        risk = "Low"
    elif probability < 0.60:
        risk = "Medium"
    else:
        risk = "High"

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4),
        "risk_level": risk
    }
