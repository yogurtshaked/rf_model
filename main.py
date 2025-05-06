from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import pickle
import uvicorn

app = FastAPI()

class Reading(BaseModel):
    date: str
    temperature: float
    humidity: float
    tds: float
    pH: float

# ─── Load your trained artifacts ───────────────────────────────────────────────
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)


# ─── Feature‐engineering helper (same as in notebook) ─────────────────────────
def create_lagged_features(df: pd.DataFrame):
    lag_features = ['Temperature', 'Humidity', 'TDS Value', 'pH Level']
    lags = [1, 2, 3, 7]

    # create shifted lags
    for feat in lag_features:
        for lag in lags:
            df[f"{feat} Lag {lag}"] = df[feat].shift(lag)

    # 7-day rolling stats
    window = 7
    for feat in lag_features:
        df[f"{feat} Rolling Mean"] = df[feat].rolling(window=window).mean()
        df[f"{feat} Rolling Std"]  = df[feat].rolling(window=window).std()

    # time features
    df['Day of Week'] = df.index.dayofweek + 1
    df['Month']       = df.index.month


# ─── Prediction endpoint ─────────────────────────────────────────────────────
@app.post("/predict")
def predict(readings: List[Reading]):
    # 1) to DataFrame
    df = pd.DataFrame([r.dict() for r in readings])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    # 2) rename to your train-time column names
    df = df.rename(columns={
        'temperature': 'Temperature',
        'humidity':    'Humidity',
        'tds':         'TDS Value',
        'pH':          'pH Level'
    })

    # 3) engineer features
    create_lagged_features(df)

    # 4) drop incomplete rows
    df_clean = df.dropna()
    if df_clean.empty:
        raise HTTPException(
            status_code=400,
            detail="Insufficient data after feature engineering; need at least 7 consecutive days."
        )

    # 5) model input & prediction
    X_raw = df_clean.copy()
    X_scaled = preprocessor.transform(X_raw)
    preds = model.predict(X_scaled)

    # 6) pick the last prediction and return it
    predicted_day = int(round(preds[-1]))
    return {"predicted_harvest_day": predicted_day}
