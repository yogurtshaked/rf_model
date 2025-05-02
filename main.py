from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
import joblib, pandas as pd
from datetime import datetime

from features import pad_to_7, create_lagged_features

preprocessor = joblib.load("preprocessor.pkl")
model        = joblib.load("model.pkl")

app = FastAPI()

class SensorData(BaseModel):
    date:        str   # "YYYY-MM-DD"
    temperature: float
    humidity:    float
    tds:         float
    ph:          float

@app.post("/predict")
def predict(window: List[SensorData]):
    # 1) Build DataFrame from your full history (Day 1 → today)
    df = pd.DataFrame([{
        "Date":        datetime.strptime(r.date, "%Y-%m-%d"),
        "Temperature": r.temperature,
        "Humidity":    r.humidity,
        "TDS Value":   r.tds,
        "pH Level":    r.ph
    } for r in window])

    # 2) If fewer than 7 days, pad Day 1 backward to get 7 rows
    df = pad_to_7(df)

    # 3) Compute lags & rolls (NaNs remain in first 6 rows)
    df_feat = create_lagged_features(df)

    # 4) Drop ANY row with a NaN → exactly your “complete” 7-day windows
    df_clean = df_feat.dropna().reset_index(drop=True)

    # 5) Grab all feature‐columns in the exact order your pipeline expects
    feat_cols = list(preprocessor.feature_names_in_)
    X_all     = preprocessor.transform(df_clean[feat_cols])

    # 6) Predict on every valid window, then average
    preds     = model.predict(X_all)
    mean_pred = float(preds.mean())

    return {
      "all_predictions":       preds.tolist(),
      "predicted_harvest_day": mean_pred
    }
