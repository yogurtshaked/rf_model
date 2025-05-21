from fastapi import FastAPI, HTTPException
from typing import List
from pydantic import BaseModel
import joblib, pandas as pd
from datetime import datetime, timedelta
from typing import Dict
import numpy as np

# Load the models
nutrient_model = joblib.load("nutrient_model.pkl")  # Dictionary of 4 RandomForestRegressors
preprocessor = joblib.load('rf_scaler.pkl')
harvest_model = joblib.load('rf_model.pkl')

app = FastAPI()

# Normal range for variables
normal_ranges = {
    'temperature': (18, 24),
    'humidity': (50, 70),
    'tds': (500, 840),
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
def create_features(
    df: pd.DataFrame,
    date_col: str = 'Date',
    phase_col: str = 'Phase'
) -> pd.DataFrame:
    """
    Build expanding & phase stats for a single time-series DataFrame.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    features = ['Temperature', 'Humidity', 'TDS Value', 'pH Level']

    # Single-phase fallback (or replace with Growth-Days logic if you add it)
    df[phase_col] = 0

    # 1) Expanding stats
    for feat in features:
        exp = df[feat].expanding(min_periods=1)
        df[f"{feat} Expanding Mean"]   = exp.mean()
        df[f"{feat} Expanding Std"]    = exp.std()
        df[f"{feat} Expanding Min"]    = exp.min()
        df[f"{feat} Expanding Max"]    = exp.max()
        df[f"{feat} Expanding Median"] = exp.median()

    # 2) Phase-based summary stats (here phase is always 0, so it's just global stats)
    agg_funcs = ['mean', 'min', 'max', 'median', 'std']
    phase_stats = (
        df.groupby(phase_col)[features]
          .agg(agg_funcs)
          .reset_index()
    )
    # flatten columns
    phase_stats.columns = (
        [phase_col] +
        [f"{feat} Phase {stat.capitalize()}"
         for feat, stat in phase_stats.columns
         if feat != phase_col]
    )
    df = df.merge(phase_stats, on=phase_col, how='left')
    return df


@app.post("/predict-harvest")
def predict_harvest(window: List[SensorData]):
    if not window:
        raise HTTPException(status_code=400, detail="Payload cannot be empty")

    # 1) Build & sort DataFrame (no early exit, no padding)
    df = pd.DataFrame([{
        'Date':        datetime.strptime(r.date, "%Y-%m-%d"),
        'Temperature': r.temperature,
        'Humidity':    r.humidity,
        'TDS Value':   r.tds,
        'pH Level':    r.ph,
    } for r in window]).sort_values('Date').reset_index(drop=True)

    # 3) Pad backwards to ensure 7 days

    # 4) Feature engineering
    df = create_features(df)
    print("\n=== Final 7-Day DataFrame ===")
    print(df)

    # 5) Prepare model input
    expected = list(preprocessor.feature_names_in_)
    last_row = df.iloc[[-1]].reindex(columns=expected, fill_value=0)
    X = preprocessor.transform(last_row)

    print("\n=== Model Input After Preprocessing ===")
    print(pd.DataFrame(X, columns=preprocessor.get_feature_names_out()))

    # 6) Predict
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
                adjustment = f"Increase TDS by {low - value:.2f}"
            elif value > high:
                adjustment = f"Decrease TDS by {value - high:.2f}"

        if clean_var == 'pH Level' and status == "Out of Range":
            if value < low:
                adjustment = f"Increase pH by {low - value:.2f}"
            elif value > high:
                adjustment = f"Decrease pH by {value - high:.2f}"

        # Only include the adjustment for TDS and pH if out of range
        results[clean_var] = {
            "value": value,  # Use the actual input value, not predicted
            "status": status,
            "adjustment": adjustment,  # Only for TDS and pH
        }

    print("Prediction Results:", results)  # Log the final results for debugging
    return results

