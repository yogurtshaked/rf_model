from fastapi import FastAPI, HTTPException
from typing import List
from pydantic import BaseModel
import joblib, pandas as pd
from datetime import datetime, timedelta
from typing import Dict

# Load the models
nutrient_model = joblib.load("nutrient_model.pkl")  # Dictionary of 4 RandomForestRegressors
preprocessor = joblib.load('preprocessor.pkl')
harvest_model = joblib.load('harvest_model.pkl')

app = FastAPI()

# Normal range for variables
normal_ranges = {
    'temperature': (18, 24),
    'humidity': (50, 70),
    'tds': (500, 1000),
    'ph': (5.5, 6.5)
}

# Model Input Data
class SensorData(BaseModel):
    date: str          # 'YYYY-MM-DD'
    temperature: float
    humidity: float
    tds: float
    ph: float

# Helper function for feature engineering
def create_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    lag_feats = ['Temperature', 'Humidity', 'TDS Value', 'pH Level']
    lags = [1, 2, 3, 7]
    window = 7

    for f in lag_feats:
        for lag in lags:
            df[f"{f} Lag {lag}"] = df[f].shift(lag)
        df[f"{f} Rolling Mean"] = df[f].rolling(window).mean()
        df[f"{f} Rolling Std"] = df[f].rolling(window).std()

    df['Day of Week'] = df['Date'].dt.dayofweek + 1
    df['Month'] = df['Date'].dt.month

    for f in lag_feats:
        for lag in lags:
            col = f"{f} Lag {lag}"
            df[col] = df[col].fillna(df[f])
        df[f"{f} Rolling Mean"] = df[f"{f} Rolling Mean"].fillna(df[f])
        df[f"{f} Rolling Std"] = df[f"{f} Rolling Std"].fillna(0)

    return df

# Harvest Day Prediction Endpoint
@app.post("/predict-harvest")
def predict_harvest(window: List[SensorData]):
    if not window:
        raise HTTPException(400, "Payload cannot be empty")

    # 1. Log original input data
    print("=== Incoming Sensor Data ===")
    for record in window:
        print(record.dict())

    # 2. Handle fewer than 7 readings (edge case)
    if len(window) < 7:
        print("Insufficient data (<7). Returning default value: 45")
        return {"predicted_harvest_day": 45}

    # 3. Prepare the DataFrame
    df = pd.DataFrame([{
        'Date': datetime.strptime(r.date, "%Y-%m-%d"),
        'Temperature': r.temperature,
        'Humidity': r.humidity,
        'TDS Value': r.tds,
        'pH Level': r.ph,
    } for r in window]).sort_values('Date').reset_index(drop=True)

    # 4. Pad the DataFrame
    while len(df) < 7:
        first = df.iloc[0].copy()
        first['Date'] -= timedelta(days=1)
        df = pd.concat([pd.DataFrame([first]), df], ignore_index=True)

    df = df.sort_values('Date').reset_index(drop=True).tail(7)

    # 5. Log the padded DataFrame
    print("\n=== Final 7-Day DataFrame ===")
    print(df)

    # 6. Feature engineering
    df = create_lagged_features(df)

    # 7. Input to model
    last_row = df[list(preprocessor.feature_names_in_)]
    X = preprocessor.transform(last_row)

    print("\n=== Model Input After Preprocessing ===")
    print(pd.DataFrame(X, columns=preprocessor.get_feature_names_out()))

    # 8. Predict
    y = harvest_model.predict(X)

    print("\n=== Harvest Day Prediction ===")
    print(int(y[0]))

    return {"predicted_harvest_day": int(y[0])}


# Nutrient Prediction Endpoint
@app.post("/predict-nutrient")
def predict_nutrients(data: SensorData) -> Dict:
    results = {}

    # For each nutrient, we check the input values against the normal range
    for clean_var, value in {
        'Temperature (°C)': data.temperature,
        'Humidity (%)': data.humidity,
        'TDS Value': data.tds,
        'pH Level': data.ph
    }.items():
        low, high = normal_ranges[clean_var.replace(" (°C)", "").replace(" (%)", "").replace(" Value", "").replace(" Level", "").lower()]

        # Check if the input value is within the normal range
        status = "Normal" if low <= value <= high else "Out of Range"
        
        # Log status for debugging
        print(f"{clean_var}: {value:.2f} is {status}")  # Log status check for debugging

        # Define adjustments only for TDS and pH
        adjustment = None
        if clean_var == 'TDS Value' and status == "Out of Range":
            if value < low:
                adjustment = f"Increase by {low - value:.2f}"
            elif value > high:
                adjustment = f"Decrease by {value - high:.2f}"

        if clean_var == 'pH Level' and status == "Out of Range":
            if value < low:
                adjustment = f"Increase by {low - value:.2f}"
            elif value > high:
                adjustment = f"Decrease by {value - high:.2f}"

        # Only include the adjustment for TDS and pH if out of range
        results[clean_var] = {
            "value": value,  # Use the actual input value, not predicted
            "status": status,
            "adjustment": adjustment,  # Only for TDS and pH
        }

    print("Prediction Results:", results)  # Log the final results for debugging
    return results

