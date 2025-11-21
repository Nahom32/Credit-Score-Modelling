from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import Dict

app = FastAPI()

# Load saved components (adjust paths if needed)
scaler = joblib.load("saved_models/standard_scaler.joblib")
model = joblib.load(
    "saved_models/xgb_model.joblib"
)  # Example: Use XGBoost; swap as needed

categorical_cols = ["Gender", "Marital Status"]  # Adjust based on your dataset
numeric_cols = [
    "Age",
    "Credit Utilization Ratio",
    "Payment History",
    "Number of Credit Accounts",
    "Loan Amount",
    "Loan Term",
]  # Adjust to match your features


# Define input schema with Pydantic (validates incoming data)
class InputData(BaseModel):
    Age: float
    Credit_Utilization_Ratio: float
    Payment_History: float
    Number_of_Credit_Accounts: float
    Loan_Amount: float
    Loan_Term: float
    Gender: str
    Marital_Status: str
    # Add more fields if your dataset has them; use snake_case for JSON compatibility


@app.post("/predict")
async def predict(data: InputData):
    try:
        # Convert Pydantic model to dict and then DataFrame
        data_dict = data.dict()
        df = pd.DataFrame([data_dict])

        # Rename columns to match training (if needed; adjust keys to underscores)
        df = df.rename(
            columns={
                "Credit_Utilization_Ratio": "Credit Utilization Ratio",
                "Payment_History": "Payment History",
                "Number_of_Credit_Accounts": "Number of Credit Accounts",
                "Loan_Amount": "Loan Amount",
                "Loan_Term": "Loan Term",
                "Marital_Status": "Marital Status",
            }
        )

        # Encode categoricals
        encoded_cats = encoder.transform(df[categorical_cols])
        encoded_df = pd.DataFrame(
            encoded_cats, columns=encoder.get_feature_names_out(categorical_cols)
        )
        df = df.drop(categorical_cols, axis=1)
        df = pd.concat([df, encoded_df], axis=1)

        # Ensure column order matches training
        df = df[numeric_cols + list(encoder.get_feature_names_out(categorical_cols))]

        # Scale numeric features
        scaled_data = scaler.transform(df)
        scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

        # Predict
        prediction = model.predict(scaled_df)

        return {"interest_rate": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Run with: uvicorn main:app --reload
