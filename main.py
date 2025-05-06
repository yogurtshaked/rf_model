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
    ph:          float

# ── feature‑engineering helper ────────────────────────────────
def create_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    lag_feats = ['Temperature', 'Humidity', 'TDS Value', 'pH Level']
    lags      = [1, 2, 3, 7]
    window    = 7

    # 1) lags & rolling stats
    for f in lag_feats:
        for lag in lags:
            df[f"{f} Lag {lag}"] = df[f].shift(lag)

        df[f"{f} Rolling Mean"] = df[f].rolling(window).mean()
        df[f"{f} Rolling Std"]  = df[f].rolling(window).std()

    # 2) calendar features
    df['Day of Week'] = df['Date'].dt.dayofweek + 1
    df['Month']       = df['Date'].dt.month

    # 3) back‑fill the NaNs **without chained assignment**
    for f in lag_feats:
        for lag in lags:                                # ← lagged columns
            col = f"{f} Lag {lag}"
            df[col] = df[col].fillna(df[f])

        roll_mean = f"{f} Rolling Mean"
        roll_std  = f"{f} Rolling Std"

        df[roll_mean] = df[roll_mean].fillna(df[f])     # mean → raw value
        df[roll_std]  = df[roll_std].fillna(0)          # std  → 0

    return df


# ── PREDICT ENDPOINT ──────────────────────────────────────────
@app.post("/predict")
def predict(window: List[SensorData]):
    if not window:
        raise HTTPException(400, "Payload cannot be empty")

    # 🟢 Early exit: fewer than 7 real readings
    if len(window) < 7:
        return {"predicted_harvest_day": 45}

    # Proceed only if valid
    df = pd.DataFrame([{
        'Date':        datetime.strptime(r.date, "%Y-%m-%d"),
        'Temperature': r.temperature,
        'Humidity':    r.humidity,
        'TDS Value':   r.tds,
        'pH Level':    r.ph,
    } for r in window]).sort_values('Date').reset_index(drop=True)

    # 🟢 Padding (only needed if you want >=7 for lags/rolling)
    while len(df) < 7:
        first = df.iloc[0].copy()
        first['Date'] -= timedelta(days=1)
        df = pd.concat([pd.DataFrame([first]), df], ignore_index=True)

    df = df.sort_values('Date').reset_index(drop=True).tail(7)

    # Featurize and predict
    df = create_lagged_features(df)

    last_row = df[list(preprocessor.feature_names_in_)]
    X = preprocessor.transform(last_row)
    y = model.predict(X)

    return {"predicted_harvest_day": int(y[0])}
