from fastapi import FastAPI, HTTPException
from typing import List
from pydantic import BaseModel
import joblib, pandas as pd
from datetime import datetime, timedelta
from typing import Dict

# Load the models
nutrient_model = joblib.load("nutrient_model.pkl")  # Dictionary of 4 RandomForestRegressors
preprocessor = joblib.load('preprocessor.pkl')
model = joblib.load('model.pkl')

app = FastAPI()

# Normal range for variables
normal_ranges = {
    'temperature': (18, 24),
    'humidity': (50, 70),
    'tds value': (500, 1000),
    'ph level': (5.5, 6.5)
}

# Model Input Data
class SensorData(BaseModel):
    date: str          # 'YYYY-MM-DD'
    temperature: float
    humidity: float
    tds: float
    ph: float

# Helper function for feature engineering
def create_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    lag_feats = ['Temperature', 'Humidity', 'TDS Value', 'pH Level']
    lags = [1, 2, 3, 7]
    window = 7

    for f in lag_feats:
        for lag in lags:
            df[f"{f} Lag {lag}"] = df[f].shift(lag)
        df[f"{f} Rolling Mean"] = df[f].rolling(window).mean()
        df[f"{f} Rolling Std"] = df[f].rolling(window).std()

    df['Day of Week'] = df['Date'].dt.dayofweek + 1
    df['Month'] = df['Date'].dt.month

    for f in lag_feats:
        for lag in lags:
            col = f"{f} Lag {lag}"
            df[col] = df[col].fillna(df[f])
        df[f"{f} Rolling Mean"] = df[f"{f} Rolling Mean"].fillna(df[f])
        df[f"{f} Rolling Std"] = df[f"{f} Rolling Std"].fillna(0)

    return df

# Harvest Day Prediction Endpoint
@app.post("/predict-harvest")
def predict_harvest(window: List[SensorData]):
    if not window:
        raise HTTPException(400, "Payload cannot be empty")

    # Check if there are fewer than 7 readings (handle edge case)
    if len(window) < 7:
        return {"predicted_harvest_day": 45}

    # Prepare the DataFrame for feature engineering
    df = pd.DataFrame([{
        'Date': datetime.strptime(r.date, "%Y-%m-%d"),
        'Temperature': r.temperature,
        'Humidity': r.humidity,
        'TDS Value': r.tds,
        'pH Level': r.ph,
    } for r in window]).sort_values('Date').reset_index(drop=True)

    # Pad the data with the first day's data (if fewer than 7 days of data)
    while len(df) < 7:
        first = df.iloc[0].copy()
        first['Date'] -= timedelta(days=1)
        df = pd.concat([pd.DataFrame([first]), df], ignore_index=True)

    df = df.sort_values('Date').reset_index(drop=True).tail(7)

    # Feature engineering and prediction
    df = create_lagged_features(df)
    last_row = df[list(preprocessor.feature_names_in_)]
    X = preprocessor.transform(last_row)
    y = model.predict(X)

    return {"predicted_harvest_day": int(y[0])}

# Nutrient Prediction Endpoint
@app.post("/predict-nutrient")
def predict_nutrients(data: SensorData) -> Dict:
    # Prepare the input for nutrient prediction
    input_df = pd.DataFrame([{
        'Temperature (°C)': data.temperature,
        'Humidity (%)': data.humidity,
        'TDS Value (ppm)': data.tds,
        'pH Level': data.ph
    }])

    results = {}

    # Predict for each nutrient using the models
    for variable, model in nutrient_model.items():
        pred = model.predict(input_df)[0]
        clean_var = variable.replace(" (°C)", "").replace(" (%)", "").replace(" (ppm)", "").replace(" Level", "")
        # Ensure 'ph' is matched correctly with 'ph level'
        if clean_var == 'ph':
            clean_var = 'ph level'
        
        low, high = normal_ranges[clean_var.lower()]

        status = "Normal" if low <= pred <= high else "Out of Range"
        adjustment = None

        if clean_var in ['TDS Value', 'pH']:
            if pred < low:
                adjustment = f"Increase by {low - pred:.2f}"
            elif pred > high:
                adjustment = f"Decrease by {pred - high:.2f}"
            else:
                adjustment = "No adjustment needed"

        results[clean_var] = {
            "predicted_value": round(pred, 2),
            "status": status,
            "adjustment": adjustment
        }

    return results
