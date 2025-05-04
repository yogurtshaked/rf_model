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
    date: str          # 'YYYY-MM-DD'
    temperature: float
    humidity:    float
    tds:         float
    pH:          float

# ── feature‑engineering helper ────────────────────────────────
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

    # back‑fill the very first rows so the model never sees NaNs
    for f in lag_feats:
        for lag in lags:
            col = f"{f} Lag {lag}"
            df[col].fillna(df[f], inplace=True)
        df[f"{f} Rolling Mean"].fillna(df[f], inplace=True)
        df[f"{f} Rolling Std"].fillna(0,      inplace=True)

    return df

# ── PREDICT ENDPOINT ──────────────────────────────────────────
@app.post("/predict")
def predict(window: List[SensorData]):
    if not window:
        raise HTTPException(400, "Payload cannot be empty")

    # 1) build DF  ✦✦✦ always parse / sort ascending ✦✦✦
    df = pd.DataFrame([{
        'Date':        datetime.strptime(r.date, "%Y-%m-%d"),
        'Temperature': r.temperature,
        'Humidity':    r.humidity,
        'TDS Value':   r.tds,
        'pH Level':    r.pH,
    } for r in window]).sort_values('Date').reset_index(drop=True)

    # 2) pad backwards (duplicate the earliest row −1 day) until len == 7
    while len(df) < 7:
        first = df.iloc[0].copy()
        first['Date'] = first['Date'] - timedelta(days=1)
        df = pd.concat([pd.DataFrame([first]), df], ignore_index=True)

    # 3) feature‑engineering
    df = create_lagged_features(df)

    # 4) take the **newest** fully‑populated row for prediction
    last_row = df.tail(1)

    # 5) keep training‑time feature order
    last_row = last_row[list(preprocessor.feature_names_in_)]

    # 6) preprocess & predict
    X  = preprocessor.transform(last_row)
    y  = model.predict(X)

    return {"predicted_harvest_day": int(y[0])}
