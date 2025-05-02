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

# Input data format
class SensorData(BaseModel):
    temperature: float
    humidity: float
    tds: float
    ph: float

@app.post('/predict')
def predict(data: SensorData):
    try:
        # Create a DataFrame from input
        input_df = pd.DataFrame([{
            'Temperature': data.temperature,
            'Humidity': data.humidity,
            'TDS Value': data.tds,
            'pH Level': data.ph
        }])

        # Debug: print input
        print("Input DF:", input_df)

        # Transform input using preprocessor
        processed_input = preprocessor.transform(input_df)

        # Debug: print processed shape
        print("Processed shape:", processed_input.shape)

        # Predict
        prediction = model.predict(processed_input)

        # Debug: print prediction
        print("Prediction:", prediction)

        # Ensure the prediction is valid
        if prediction is not None and len(prediction) > 0:
            return {"predicted_harvest_day": int(prediction[0])}
        else:
            return {"error": "Predicted harvest day is missing or invalid."}

    except Exception as e:
        print("Exception:", str(e))
        return {"error": str(e)}
