from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# === Load trained model and scaler ===
scaler = joblib.load('scaler_X.pkl')
model = joblib.load('model.pkl')
feature_names = joblib.load('feature_names.pkl')

# === Initialize FastAPI ===
app = FastAPI()

# === Request schema ===
class SensorData(BaseModel):
    temperature: float
    humidity: float
    tds: float
    ph: float
    date: str  # Format: 'YYYY-MM-DD'

# === Feature engineering: lags, rolling stats, time components ===
def create_features_from_input(data_df):
    df = data_df.copy()
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month

    for col in ['Temperature', 'Humidity', 'TDS Value', 'pH Level']:
        for lag in [1, 2, 3, 7]:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        df[f'{col}_roll_mean'] = df[col].rolling(window=7).mean()
        df[f'{col}_roll_std']  = df[col].rolling(window=7).std()

    return df.dropna()

# === Prediction endpoint ===
@app.post('/predict')
def predict(data: SensorData):
    try:
        # Create 10-day history with constant values to allow lag/rolling features
        date = datetime.strptime(data.date, '%Y-%m-%d')
        date_range = pd.date_range(end=date, periods=10)

        df = pd.DataFrame({
            'Date': date_range,
            'Temperature': [data.temperature] * 10,
            'Humidity': [data.humidity] * 10,
            'TDS Value': [data.tds] * 10,
            'pH Level': [data.ph] * 10
        }).set_index('Date')

        # Generate features
        features_df = create_features_from_input(df)

        if features_df.empty:
            return {"error": "Not enough data to compute lag/rolling features."}

        latest_row = features_df.tail(1)

        # Match feature order to training
        latest_row = latest_row[feature_names]

        # Scale and predict
        X_scaled = scaler.transform(latest_row)
        prediction = model.predict(X_scaled)

        return {"predicted_harvest_day": int(round(prediction[0]))}

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}
