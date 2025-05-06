from fastapi import FastAPI, HTTPException
from typing import List
from pydantic import BaseModel
import joblib, pandas as pd
from datetime import datetime, timedelta

# ── load artifacts ────────────────────────────────────────────
preprocessor = joblib.load('preprocessor.pkl')
model        = joblib.load('model.pkl')

app = FastAPI()

class SensorData(BaseModel):
    date: str
    temperature: float
    humidity: float
    tds: float
    ph: float

# ── Feature engineering ───────────────────────────────────────
def create_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    lag_feats = ['Temperature', 'Humidity', 'TDS Value', 'pH Level']
    lags      = [1, 2, 3, 7]
    window    = 7

    for f in lag_feats:
        for lag in lags:
            df[f"{f} Lag {lag}"] = df[f].shift(lag)

        df[f"{f} Rolling Mean"] = df[f].rolling(window).mean()
        df[f"{f} Rolling Std"]  = df[f].rolling(window).std()

    df['Day of Week'] = df['Date'].dt.dayofweek + 1
    df['Month']       = df['Date'].dt.month

    df = df.fillna(method='ffill').fillna(method='bfill')  # better than raw overwrite

    return df

# ── PREDICT endpoint ──────────────────────────────────────────
@app.post("/predict")
def predict(window: List[SensorData]):
    if not window:
        raise HTTPException(400, "Payload cannot be empty")

    if len(window) < 7:
        # Fallback logic: estimate based on days since planting
        first_date = datetime.strptime(window[0].date, "%Y-%m-%d")
        today = datetime.today()
        elapsed_days = (today - first_date).days
        estimated_total = 35  # e.g. lettuce (Kratky); make dynamic if needed

        return {
            "prediction_method": "fallback",
            "days_elapsed": elapsed_days,
            "estimated_total_growth_days": estimated_total,
            "predicted_harvest_day": max(estimated_total - elapsed_days, 5)
        }

    # Convert to DataFrame
    df = pd.DataFrame([{
        'Date':        datetime.strptime(r.date, "%Y-%m-%d"),
        'Temperature': r.temperature,
        'Humidity':    r.humidity,
        'TDS Value':   r.tds,
        'pH Level':    r.ph,
    } for r in window])

    df = df.sort_values('Date').reset_index(drop=True)
    df = create_lagged_features(df)

    # Use the most recent day for prediction
    last_row = df.iloc[-1:]
    X = preprocessor.transform(last_row[preprocessor.feature_names_in_])
    y = model.predict(X)

    predicted_days_remaining = int(y[0])
    today = datetime.today()
    predicted_harvest_date = today + timedelta(days=predicted_days_remaining)

    return {
        "prediction_method": "model",
        "predicted_days_remaining": predicted_days_remaining,
        "predicted_harvest_date": predicted_harvest_date.strftime("%Y-%m-%d")
    }
