from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime, timedelta

# --- load your artifacts ---
preprocessor = joblib.load('preprocessor.pkl')
model        = joblib.load('model.pkl')

app = FastAPI()

class SensorData(BaseModel):
    date: str        # 'YYYY-MM-DD'
    temperature: float
    humidity:    float
    tds:         float
    ph:          float

def create_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    lag_feats = ['Temperature', 'Humidity', 'TDS Value', 'pH Level']
    lags      = [1, 2, 3, 7]
    window    = 7

    # compute raw shifts & rolls
    for f in lag_feats:
        for lag in lags:
            df[f"{f} Lag {lag}"] = df[f].shift(lag)
        df[f"{f} Rolling Mean"] = df[f].rolling(window).mean()
        df[f"{f} Rolling Std"]  = df[f].rolling(window).std()

    # time features
    df['Day of Week'] = df['Date'].dt.dayofweek + 1
    df['Month']       = df['Date'].dt.month

    # fill any NaNs (should only occur for the first few rows)
    for f in lag_feats:
        for lag in lags:
            col = f"{f} Lag {lag}"
            df[col].fillna(df[f], inplace=True)
        df[f"{f} Rolling Mean"].fillna(df[f], inplace=True)
        df[f"{f} Rolling Std"].fillna(0, inplace=True)

    return df

@app.post("/predict")
def predict(window: List[SensorData]):
    # 1) build DF
    records = []
    for rec in window:
        records.append({
            'Date':        datetime.strptime(rec.date, "%Y-%m-%d"),
            'Temperature': rec.temperature,
            'Humidity':    rec.humidity,
            'TDS Value':   rec.tds,
            'pH Level':    rec.ph
        })
    df = pd.DataFrame(records).sort_values('Date').reset_index(drop=True)

    # 2) pad backward to 7 days if needed
    if len(df) < 7:
        first = df.iloc[0]
        while len(df) < 7:
            first = first.copy()
            first['Date'] = first['Date'] - timedelta(days=1)
            df = pd.concat([pd.DataFrame([first]), df], ignore_index=True)
        df = df.sort_values('Date').reset_index(drop=True)

    # 3) feature-engineer
    df = create_lagged_features(df)

    # 4) pick the last fully-populated row
    last_row = df.dropna().tail(1)

    # ---- DEBUG LOGGING ----
    # 4a) raw features
    raw_feat = last_row.to_dict(orient="records")[0]
    print("=== RAW LAST ROW ===")
    print(raw_feat)

    # 4b) enforce training column order
    cols = list(preprocessor.feature_names_in_)
    last_row = last_row[cols]
    print("=== FEATURE ORDER BEFORE TRANSFORM ===")
    print(last_row.to_dict(orient="records")[0])

    # 4c) after transform
    processed = preprocessor.transform(last_row)
    print("=== AFTER PREPROCESSOR.TRANSFORM ===")
    print(processed)
    # ------------------------

    # 5) predict
    y = model.predict(processed)
    print("=== MODEL PREDICTION ===", y)

    return {"predicted_harvest_day": int(y[0])}
