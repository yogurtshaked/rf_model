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
    # --- build & pad & featurize as before ---
    df = pd.DataFrame([{
        "Date":        datetime.strptime(r.date, "%Y-%m-%d"),
        "Temperature": r.temperature,
        "Humidity":    r.humidity,
        "TDS Value":   r.tds,
        "pH Level":    r.ph
    } for r in window])

    df = pad_to_7(df)
    df_feat = create_lagged_features(df)

    # --- now drop incomplete windows (as in your notebook) ---
    max_lag = 7
    if len(df_feat) > max_lag:
        df_windows = df_feat.iloc[max_lag:].reset_index(drop=True)
    else:
        df_windows = df_feat.tail(1).reset_index(drop=True)

    # --- enforce your pipeline's column order ---
    expected_cols = list(preprocessor.feature_names_in_)
    actual_cols   = list(df_windows.columns)

    # 1) check for missing / extra columns
    missing = set(expected_cols) - set(actual_cols)
    extra   = set(actual_cols) - set(expected_cols)
    if missing:
        print("❌ Missing columns:", missing)
    if extra:
        print("❌ Extra columns:", extra)
    if not missing and not extra:
        print("✅ Columns match exactly.")

    # 2) select & print values
    last_row = df_windows[expected_cols].tail(1)
    values = last_row.to_dict(orient="records")[0]
    print("=== FINAL FEATURE VECTOR DUMP ===")
    for col, val in values.items():
        print(f"  • {col}: {val!r}")

    # 3) transform & predict
    X_all = preprocessor.transform(last_row)
    print("=== SCALED VECTOR ===", X_all)
    preds = model.predict(X_all)
    print("=== RAW MODEL PREDICTIONS ===", preds)

    # 4) average & return
    mean_pred = float(preds.mean())
    return {
      "all_predictions":       preds.tolist(),
      "predicted_harvest_day": mean_pred
    }



