from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime

# Load preprocessor and model
preprocessor = joblib.load('preprocessor.pkl')
model = joblib.load('model.pkl')

# Initialize FastAPI app
app = FastAPI()

# Input data format
class SensorData(BaseModel):
    temperature: float
    humidity: float
    tds: float
    ph: float
    date: str  # 'YYYY-MM-DD'

# Feature engineering function for a single-row input
def add_features(df):
    # Convert date to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"])

    # Time-based features
    df['Day of Week'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month

    # Lag and rolling features (set to NaN for first row)
    feature_columns = ['Temperature', 'Humidity', 'TDS Value', 'pH Level']
    for col in feature_columns:
        for lag in [1, 2, 3, 7]:
            df[f'{col} Lag {lag}'] = pd.NA  # or float('nan')
        df[f'{col} Rolling Mean'] = pd.NA
        df[f'{col} Rolling Std'] = pd.NA

    return df

# Prediction endpoint
@app.post('/predict')
def predict(data: SensorData):
    try:
        # Create initial DataFrame from input
        input_data = pd.DataFrame([{
            'Temperature': data.temperature,
            'Humidity': data.humidity,
            'TDS Value': data.tds,
            'pH Level': data.ph,
            'Date': datetime.strptime(data.date, '%Y-%m-%d')
        }])

        print("Raw Input:", input_data)

        # Add lag, rolling, and time features
        input_data = add_features(input_data)

        print("With Features:", input_data)

        # Pass through the preprocessor
        processed_input = preprocessor.transform(input_data)

        print("Processed Input Shape:", processed_input.shape)

        # Make prediction
        prediction = model.predict(processed_input)

        print("Prediction:", prediction)

        # Return result
        if prediction is not None and len(prediction) > 0:
            return {"predicted_harvest_day": int(prediction[0])}
        else:
            return {"error": "Prediction is missing or invalid."}

    except Exception as e:
        print("Exception:", str(e))
        return {"error": str(e)}
