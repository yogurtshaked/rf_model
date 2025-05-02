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
    # 1) Build raw DF
    recs = [{
        "Date":        datetime.strptime(r.date, "%Y-%m-%d"),
        "Temperature": r.temperature,
        "Humidity":    r.humidity,
        "TDS Value":   r.tds,
        "pH Level":    r.ph
    } for r in window]
    df = pd.DataFrame(recs)

    # 2) Pad up to 7 days if needed, then featurize
    df = pad_to_7(df)
    df = create_lagged_features(df)

    # 3) Drop windows with NaNs → one row per valid 7-day window
    df_clean = df.dropna().reset_index(drop=True)

    # 4) Enforce the exact column order your pipeline expects
    feat_cols = list(preprocessor.feature_names_in_)
    X_all     = preprocessor.transform(df_clean[feat_cols])

    # 5) Predict on every window, then take the mean
    preds      = model.predict(X_all)
    mean_pred  = float(preds.mean())

    return {
      "all_predictions":       preds.tolist(),
      "predicted_harvest_day": mean_pred
    }
