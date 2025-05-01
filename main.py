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
    temperature: float
    humidity: float
    tds: float
    ph: float

# Define prediction endpoint
@app.post('/predict')
def predict(data: SensorData):
    try:
        # Prepare input data
        input_data = np.array([[data.temperature, data.humidity, data.tds, data.ph]])
        # Perform prediction
        prediction = model.predict(input_data)
        return {"predicted_day": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}

# For local and Render deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
