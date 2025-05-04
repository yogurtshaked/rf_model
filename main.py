import FastAPI
from typing import List
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime

from features import create_lagged_features, pad_to_7

# load artifacts
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
    # build raw DF
    records = [{
        "Date":        datetime.strptime(r.date, "%Y-%m-%d"),
        "Temperature": r.temperature,
        "Humidity":    r.humidity,
        "TDS Value":   r.tds,
        "pH Level":    r.ph
    } for r in window]
    df = pd.DataFrame(records)

    # pad & featurize
    df = pad_to_7(df)
    df = create_lagged_features(df)

    # pick the last fully‐populated row
    last_row = df.dropna().tail(1)

    # enforce the exact column order your pipeline saw at train time
    feature_cols = list(preprocessor.feature_names_in_)
    last_row     = last_row[feature_cols]

    # debug logs (optional)
    print("FEATURE VECTOR:", last_row.to_dict(orient="records")[0])
    X = preprocessor.transform(last_row)
    print("TRANSFORMED VECTOR:", X)
    y = model.predict(X)
    print("PREDICTION:", y)

    return {"predicted_harvest_day": int(y[0])}
