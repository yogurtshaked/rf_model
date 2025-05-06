from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import joblib
import datetime

app = FastAPI()

# Load trained model and preprocessor (e.g., a scaler)
model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")  # e.g., StandardScaler or full pipeline

class Reading(BaseModel):
    date: str
    temperature: float
    humidity: float
    tds: float
    pH: float

@app.post("/predict")
def predict_growth_days(readings: List[Reading]):
    if len(readings) < 7:
        return {"predicted_harvest_day": 45}

    # Convert to DataFrame
    df = pd.DataFrame([r.dict() for r in readings])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df.sort_index()

    # Rename to match feature names in training
    df.rename(columns={
        'temperature': 'Temperature',
        'humidity': 'Humidity',
        'tds': 'TDS Value',
        'pH': 'pH Level'
    }, inplace=True)

    # Feature engineering
    df = create_lagged_features(df)

    # Drop rows with NaNs due to lag/rolling features
    df_clean = df.dropna()

    if df_clean.empty:
        return {"predicted_harvest_day": 45}

    # Only predict the last date (like a snapshot prediction)
    X = df_clean.iloc[[-1]]  # Last row only
    X_scaled = preprocessor.transform(X)  # Match preprocessing used in training

    prediction = model.predict(X_scaled)[0]
    return {"predicted_harvest_day": int(prediction)}

def create_lagged_features(lettuce_df):
    lag_features = ['Temperature', 'Humidity', 'TDS Value', 'pH Level']
    lags = [1, 2, 3, 7]

    for feature in lag_features:
        for lag in lags:
            lettuce_df[f"{feature} Lag {lag}"] = lettuce_df[feature].shift(lag)

    window = 7
    for feature in lag_features:
        lettuce_df[f"{feature} Rolling Mean"] = lettuce_df[feature].rolling(window=window).mean()
        lettuce_df[f"{feature} Rolling Std"] = lettuce_df[feature].rolling(window=window).std()

    lettuce_df['Day of Week'] = lettuce_df.index.dayofweek + 1
    lettuce_df['Month'] = lettuce_df.index.month

    return lettuce_df
