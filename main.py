from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the pre-trained Random Forest model from model.pkl
model = joblib.load('model.pkl')

# Initialize the FastAPI app
app = FastAPI()

# Define input data format using Pydantic
class SensorData(BaseModel):
    temperature: float
    humidity: float
    tds: float
    ph: float

# Define prediction endpoint
@app.post("/predict")
def predict(data: SensorData):
    # Extract data from the request
    input_data = np.array([[data.temperature, data.humidity, data.tds, data.ph]])

    # Make prediction using the Random Forest model
    prediction = model.predict(input_data)

    # Return the predicted harvest day
    return {"predicted_harvest_day": int(prediction[0])}
