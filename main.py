from fastapi import FastAPI, HTTPException
from typing  import List
from pydantic import BaseModel
import joblib, pandas as pd
from datetime import datetime, timedelta

preprocessor = joblib.load('preprocessor.pkl')
model        = joblib.load('model.pkl')
app = FastAPI()

class SensorData(BaseModel):
    date:        str    # 'YYYY-MM-DD'
    temperature: float
    humidity:    float
    tds:         float
    pH:          float

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

    # back-fill NaNs
    for f in lag_feats:
        for lag in lags:
            col = f"{f} Lag {lag}"
            df[col] = df[col].fillna(df[f])
        df[f"{f} Rolling Mean"] = df[f"{f} Rolling Mean"].fillna(df[f])
        df[f"{f} Rolling Std"]  = df[f"{f} Rolling Std"].fillna(0)

    return df

@app.post("/predict")
def predict(window: List[SensorData]):
    if not window:
        raise HTTPException(400, "No data provided")

    # 1) Build & sort
    df = pd.DataFrame([{
        "Date":        datetime.strptime(r.date, "%Y-%m-%d"),
        "Temperature": r.temperature,
        "Humidity":    r.humidity,
        "TDS Value":   r.tds,
        "pH Level":    r.pH
    } for r in window]).sort_values("Date").reset_index(drop=True)

    # 2) Pad so shift(7) has real values
    df = pad_to_minimum(df)

    # 3) Create lags + rolls (leave NaNs!)
    df_feat = create_lagged_features(df)

    # 4) Drop the first 7 rows (incomplete windows)
    df_clean = df_feat.dropna().reset_index(drop=True)

    # 5) Take the last row only
    cols     = list(preprocessor.feature_names_in_)
    last_row = df_clean.tail(1)[cols]

    # 6) Transform & predict
    X = preprocessor.transform(last_row)
    y = model.predict(X)

    return {"predicted_harvest_day": int(y[0])}
