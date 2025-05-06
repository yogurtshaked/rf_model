from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import joblib
import os

# Load pre-trained model and preprocessor
model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

app = FastAPI()

# Define the structure of input data
class Reading(BaseModel):
    date: str
    temperature: float
    humidity: float
    tds: float
    pH: float

@app.post("/predict")
def predict(readings: List[Reading]):
    try:
        # Convert readings to a DataFrame
        df = pd.DataFrame([{
            "Temperature": float(r.temperature),
            "Humidity": float(r.humidity),
            "TDS Value": float(r.tds),
            "pH Level": float(r.pH),
            "date": pd.to_datetime(r.date)
        } for r in readings])

        df = df.sort_values(by="date")
        df.set_index("date", inplace=True)

        # Create lag and rolling features
        df["lag_temp"] = df["Temperature"].shift(1)
        df["lag_humidity"] = df["Humidity"].shift(1)
        df["lag_tds"] = df["TDS Value"].shift(1)
        df["lag_ph"] = df["pH Level"].shift(1)

        df["roll_temp"] = df["Temperature"].rolling(window=3).mean()
        df["roll_humidity"] = df["Humidity"].rolling(window=3).mean()
        df["roll_tds"] = df["TDS Value"].rolling(window=3).mean()
        df["roll_ph"] = df["pH Level"].rolling(window=3).mean()

        # Drop rows with NaN values from lag/rolling ops
        df_clean = df.dropna()

        # Extract the latest row for prediction
        X = df_clean.iloc[[-1]]

        # Debug logs for feature checking
        print("Predicting with input:")
        print(X)
        print("Columns:", X.columns.tolist())
        print("NaNs:", X.isna().sum())

        # Transform with preprocessor and predict
        X_scaled = preprocessor.transform(X)
        prediction = model.predict(X_scaled)[0]

        return {"predicted_harvest_day": int(prediction)}
    
    except Exception as e:
        print("Error during prediction:", str(e))
        return {"error": "Prediction failed. Check input format or model compatibility."}
