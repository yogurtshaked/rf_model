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
    # build DataFrame
    df = pd.DataFrame([{
        "Date":        datetime.strptime(r.date, "%Y-%m-%d"),
        "Temperature": r.temperature,
        "Humidity":    r.humidity,
        "TDS Value":   r.tds,
        "pH Level":    r.ph
    } for r in window])

    # pad → featurize
    df = pad_to_7(df)
    df_feat = create_lagged_features(df)

    # drop the first max_lag rows (just like dropna in the notebook)
    max_lag = 7
    if len(df_feat) > max_lag:
        df_windows = df_feat.iloc[max_lag:].reset_index(drop=True)
    else:
        df_windows = df_feat.tail(1).reset_index(drop=True)

    # select & transform all windows
    cols = preprocessor.feature_names_in_
    X_all = preprocessor.transform(df_windows[cols])

    # predict + average
    preds = model.predict(X_all)
    return {
      "all_predictions": preds.tolist(),
      "predicted_harvest_day": float(preds.mean())
    }

