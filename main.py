from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from typing import List

# Load your trained model
model = joblib.load("harvest_model.pkl")

app = FastAPI()

# Pydantic schema for input
class Reading(BaseModel):
    date: str
    temperature: float
    humidity: float
    tds: float
    pH: float

class PredictionResponse(BaseModel):
    predicted_harvest_day: int


def create_lagged_features(df):
    df = df.copy()
    lag_features = ['Temperature', 'Humidity', 'TDS Value', 'pH Level']
    lags = [1, 2, 3, 7]
    
    for feature in lag_features:
        for lag in lags:
            df[f"{feature} Lag {lag}"] = df[feature].shift(lag)

    window = 7
    for feature in lag_features:
        df[f"{feature} Rolling Mean"] = df[feature].rolling(window=window).mean()
        df[f"{feature} Rolling Std"] = df[feature].rolling(window=window).std()

    df['Day of Week'] = df.index.dayofweek + 1
    df['Month'] = df.index.month
    return df


@app.post("/predict", response_model=PredictionResponse)
def predict(readings: List[Reading]):
    if len(readings) < 7:
        return {"predicted_harvest_day": 45}

    # Convert input to DataFrame
    try:
        df = pd.DataFrame([r.dict() for r in readings])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')  # Ensure chronological order
        df.set_index('date', inplace=True)
        df.rename(columns={
            'temperature': 'Temperature',
            'humidity': 'Humidity',
            'tds': 'TDS Value',
            'pH': 'pH Level'
        }, inplace=True)

        # Feature engineering
        df_feat = create_lagged_features(df)
        df_feat = df_feat.dropna()

        if df_feat.empty:
            raise ValueError("Not enough data after feature processing.")

        # Prepare model input (drop target column if any)
        latest_row = df_feat.iloc[[-1]]  # Only last day
        X = latest_row  # Ensure features match training
        prediction = model.predict(X)[0]
        return {"predicted_harvest_day": round(float(prediction))}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
