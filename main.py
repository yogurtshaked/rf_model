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

def create_lagged_features(input_df):
    lag_features = ['Temperature', 'Humidity', 'TDS Value', 'pH Level']
    lags = [1, 2, 3, 7]
    window = 7

    # 1) compute shifts + rolling
    for feature in lag_features:
        for lag in lags:
            input_df[f"{feature} Lag {lag}"] = input_df[feature].shift(lag)
        input_df[f"{feature} Rolling Mean"] = input_df[feature].rolling(window=window).mean()
        input_df[f"{feature} Rolling Std"]  = input_df[feature].rolling(window=window).std()

    input_df['Day of Week'] = input_df['Date'].dt.dayofweek + 1
    input_df['Month']       = input_df['Date'].dt.month

    # 2) fill all the NaNs with sensible defaults
    for feature in lag_features:
        for lag in lags:
            col = f"{feature} Lag {lag}"
            input_df[col].fillna(input_df[feature], inplace=True)
        mean_col = f"{feature} Rolling Mean"
        std_col  = f"{feature} Rolling Std"
        input_df[mean_col].fillna(input_df[feature], inplace=True)
        input_df[std_col].fillna(0, inplace=True)

    return input_df

@app.post('/predict')
def predict(data: SensorData):
    try:
        input_df = pd.DataFrame([{
            'Date':  datetime.strptime(data.date, '%Y-%m-%d'),
            'Temperature': data.temperature,
            'Humidity':    data.humidity,
            'TDS Value':   data.tds,
            'pH Level':    data.ph
        }])
        input_df['Date'] = pd.to_datetime(input_df['Date'])

        # build all features and impute
        full_df = create_lagged_features(input_df)

        # now we know full_df has no NaNs, so we can skip dropna()
        processed_input = preprocessor.transform(full_df)
        prediction      = model.predict(processed_input)

        return {"predicted_harvest_day": int(prediction[0])}

    except Exception as e:
        print("Exception:", e)
        return {"error": str(e)}

