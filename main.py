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

        df = df.dropna(subset=[roll_mean])
        df = df.dropna(subset=[roll_std])         # std  → 0

    return df


# ── PREDICT ENDPOINT ──────────────────────────────────────────
@app.post("/predict")
def predict(window: List[SensorData]):
    if not window:
        raise HTTPException(400, "Payload cannot be empty")

    # 1) Hard-code 45 if fewer than 7 real readings
    if len(window) < 7:
        return {"predicted_harvest_day": 45}

    # 2) Build + pad to at least 7 days
    df = pd.DataFrame([...]) \
           .sort_values("Date") \
           .reset_index(drop=True)
    while len(df) < 7:
        first = df.iloc[0].copy()
        first["Date"] -= timedelta(days=1)
        df = pd.concat([pd.DataFrame([first]), df], ignore_index=True)

    # 3) Now feature-engineer the entire history
    df = create_lagged_features(df)

    # 4) Take only the last row for *today’s* prediction
    last_row = df.tail(1)[list(preprocessor.feature_names_in_)]

    # 5) Transform & predict
    X = preprocessor.transform(last_row)
    y = model.predict(X)

    return {"predicted_harvest_day": int(y[0])}

