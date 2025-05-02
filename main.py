from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Load preprocessor and model
preprocessor = joblib.load('preprocessor.pkl')
model = joblib.load('model.pkl')

# Initialize FastAPI app
app = FastAPI()

# Input data format
class SensorData(BaseModel):
    date: str
    temperature: float
    humidity: float
    tds: float
    ph: float

@app.post('/predict')
def predict(data: SensorData):
    try:
        # Create a DataFrame from input (only the user-provided features)
        input_df = pd.DataFrame([{
            'Date': datetime.strptime(data.date, '%Y-%m-%d'),
            'Temperature': data.temperature,
            'Humidity': data.humidity,
            'TDS Value': data.tds,
            'pH Level': data.ph
        }])

        # Debug: print the input data
        print("Input DF:", input_df)
        
        # Convert 'Date' column to datetime format to extract time-based features
        input_df['Date'] = pd.to_datetime(input_df['Date'])

        # Add placeholder columns for missing features (lagged and rolling features)
        lag_features = ['Temperature', 'Humidity', 'TDS Value', 'pH Level']
        lags = [1, 2, 3, 7]  # These lags were used during model training
    
        # Add placeholder lag features (use the .shift() method to create lag features)
        for feature in lag_features:
            for lag in lags:
                input_df[f"{feature} Lag {lag}"] = input_df[feature].shift(lag)

        # Add placeholder rolling stats (mean and std)
        window = 7
        for feature in lag_features:
            input_df[f"{feature} Rolling Mean"] = input_df[feature].rolling(window=window).mean()
            input_df[f"{feature} Rolling Std"] = input_df[feature].rolling(window=window).std()

        # Add time-based features (Day of Week, Month) derived from 'Date'
        input_df['Day of Week'] = input_df['Date'].dt.dayofweek + 1  # Monday = 1, Sunday = 7
        input_df['Month'] = input_df['Date'].dt.month  # Extract month (1-12)

        # Debug: print the updated DataFrame
        print("Updated Input DF with placeholders:", input_df)

        # Transform the input using the preprocessor (it expects 30 features)
        processed_input = preprocessor.transform(input_df)
        print("Processed Input:", processed_input)

        # Predict using the model
        prediction = model.predict(processed_input)

        # Debug: print prediction
        print("Prediction:", prediction)

        # Ensure the prediction is valid
        if prediction is not None and len(prediction) > 0:
            return {"predicted_harvest_day": int(prediction[0])}
        else:
            return {"error": "Predicted harvest day is missing or invalid."}

    except Exception as e:
        # Capture and log the error message
        print("Exception:", str(e))
        return {"error": str(e)}
