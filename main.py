from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime

from features import pad_to_7, create_lagged_features

# --- load your artifacts ---
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
    # 1) build raw DF
    records = [{
        "Date":        datetime.strptime(r.date, "%Y-%m-%d"),
        "Temperature": r.temperature,
        "Humidity":    r.humidity,
        "TDS Value":   r.tds,
        "pH Level":    r.ph
    } for r in window]
    df = pd.DataFrame(records)

    # 2) pad to 7 days if too short
    df = pad_to_7(df)

    # 3) compute all features
    df = create_lagged_features(df)

    # 4) drop the NaN-rows (first 6 windows), keep one row per 7-day window
    df_clean = df.dropna().reset_index(drop=True)

    # 5) select all the feature-columns in the exact order the preprocessor expects
    feature_cols = list(preprocessor.feature_names_in_)
    X_all        = preprocessor.transform(df_clean[feature_cols])

    # 6) predict on every sliding‐window
    preds = model.predict(X_all)

    # 7) return both the full list and their mean
    return {
      "all_predictions":       preds.tolist(),
      "predicted_harvest_day": float(preds.mean())
    }
