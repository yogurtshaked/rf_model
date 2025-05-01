from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os
import uvicorn

# Load the pre-trained model
model = joblib.load('model.pkl')

# Initialize the FastAPI app
app = FastAPI()

# Define input format
class SensorData(BaseModel):
    date: str  # make sure your frontend sends this!
    temperature: float
    humidity: float
    tds: float
    ph: float

# Prediction endpoint
@app.post('/predict')
def predict(data: SensorData):
    input_df = pd.DataFrame([{
        'Date': data.date,
        'Temperature': data.temperature,
        'Humidity': data.humidity,
        'TDS Value': data.tds,
        'pH Level': data.ph
    }])

    prediction = pipeline.predict(input_df)
    return {'predicted_day': int(prediction[0])}

# For local and Render deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
