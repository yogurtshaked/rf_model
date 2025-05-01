from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os
import uvicorn

# Load the pre-trained model
model = joblib.load('model.pkl')

preprocess_data = joblib.load('preprocessing.pkl')

# Initialize the FastAPI app
app = FastAPI()

# Define input format
class SensorData(BaseModel):
    date: str  # make sure your frontend sends this!
    temperature: float
    humidity: float
    tds: float
    ph: float

@app.post('/predict')
def predict(data: SensorData):
    # Create DataFrame from input data
    input_data = pd.DataFrame([data.dict()])

    # Apply preprocessing
    processed_data = preprocess_data(input_data)

    # Extract features (make sure to match training feature names)
    features = processed_data[['Temperature', 'Humidity', 'TDS Value', 'pH Level', 'Temperature Lag 1',
                               'Temperature Lag 2', 'Temperature Lag 3', 'Temperature Lag 7',
                               'Humidity Lag 1', 'Humidity Lag 2', 'Humidity Lag 3', 'Humidity Lag 7',
                               'TDS Value Lag 1', 'TDS Value Lag 2', 'TDS Value Lag 3',
                               'TDS Value Lag 7', 'pH Level Lag 1', 'pH Level Lag 2', 'pH Level Lag 3',
                               'pH Level Lag 7', 'Temperature Rolling Mean', 'Temperature Rolling Std',
                               'Humidity Rolling Mean', 'Humidity Rolling Std',
                               'TDS Value Rolling Mean', 'TDS Value Rolling Std',
                               'pH Level Rolling Mean', 'pH Level Rolling Std', 'Day of Week', 'Month']]

    # Make prediction
    prediction = model.predict(features)

    # Return predicted harvest day
    return {"predicted_harvest_day": int(prediction[0])}

# For local and Render deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
