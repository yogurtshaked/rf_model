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
    date: str  # Date in 'YYYY-MM-DD' format

@app.post('/predict')
def predict(data: SensorData):
    try:
        # Create a DataFrame from the user input, including the date
        input_data = pd.DataFrame([{
            'Temperature': data.temperature,
            'Humidity': data.humidity,
            'TDS Value': data.tds,
            'pH Level': data.ph,
            'Date': datetime.strptime(data.date, '%Y-%m-%d')  # Convert string to datetime
        }])

        # Debug: print input
        print("Input DF:", input_data)

        # Pass the data through the preprocessor to generate features (lags, rolling, time-based)
        processed_input = preprocessor.transform(input_data)

        # Debug: print processed shape
        print("Processed shape:", processed_input.shape)

        # Make the prediction using the preprocessed data
        prediction = model.predict(processed_input)

        # Debug: print the prediction
        print("Prediction:", prediction)

        # Ensure the prediction is valid and return the result
        if prediction is not None and len(prediction) > 0:
            return {"predicted_harvest_day": int(prediction[0])}
        else:
            return {"error": "Predicted harvest day is missing or invalid."}

    except Exception as e:
        print("Exception:", str(e))
        return {"error": str(e)}

