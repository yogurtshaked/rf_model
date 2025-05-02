# main.py
from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
import joblib, pandas as pd
from datetime import datetime

from features import pad_to_minimum, create_lagged_features, LAGS

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
    df = pd.DataFrame([{
        "Date":        datetime.strptime(r.date, "%Y-%m-%d"),
        "Temperature": r.temperature,
        "Humidity":    r.humidity,
        "TDS Value":   r.tds,
        "pH Level":    r.ph
    } for r in window])

    # 2) pad until at least LAGS+1 rows
    df = pad_to_minimum(df)

    # 3) featurize (no fill-na here)
    df_feat = create_lagged_features(df)

    # 4) drop the first max(LAGS) rows (incomplete windows)
    df_windows = df_feat.iloc[len(LAGS):].reset_index(drop=True)

    # 5) now df_windows[feat_cols] has at least one row
    feat_cols = list(preprocessor.feature_names_in_)
    X_all     = preprocessor.transform(df_windows[feat_cols])

    # 6) predict on each window & average
    preds     = model.predict(X_all)
    mean_pred = float(preds.mean())

    return {
      "all_predictions":       preds.tolist(),
      "predicted_harvest_day": mean_pred
    }
