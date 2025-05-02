from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load preprocessor and model
preprocessor = joblib.load('preprocessor.pkl')
model = joblib.load('model.pkl')

# Initialize FastAPI app
app = FastAPI()

# Input format
class SensorData(BaseModel):
    temperature: float
    humidity: float
    tds: float
    ph: float

@app.post('/predict')
def predict(data: SensorData):
    try:
        # Create DataFrame from input
        input_df = pd.DataFrame([{
            'Temperature': data.temperature,
            'Humidity': data.humidity,
            'TDS Value': data.tds,
            'pH Level': data.ph
        }])

        # Apply preprocessing to match model input format
        processed_input = preprocessor.transform(input_df)

        # Predict using model
        prediction = model.predict(processed_input)

        return {"predicted_harvest_day": int(prediction[0])}
    
    except Exception as e:
        return {"error": str(e)}
