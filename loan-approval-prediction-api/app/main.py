from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Loan Approval Prediction API")

# Load trained pipeline
model = joblib.load("model/loan_pipeline.pkl")


class LoanApplication(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str


@app.get("/")
def home():
    return {
        "message": "Loan Approval Prediction API is running"
    }


@app.post("/predict")
def predict(application: LoanApplication):
    input_data = pd.DataFrame([application.dict()])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    result = "Approved" if prediction == 1 else "Rejected"

    return {
        "prediction": result,
        "approval_probability": round(float(probability), 4)
    }