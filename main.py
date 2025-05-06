import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
from datetime import datetime

# Load preprocessor and model artifacts
preprocessor = joblib.load('preprocessor.pkl')
model = joblib.load('model.pkl')

app = FastAPI()

class SensorData(BaseModel):
    date: str
    temperature: float
    humidity: float
    tds: float
    ph: float


def create_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with at least 7 rows of chronological data,
    compute lagged and rolling statistics, fill NaNs with sensible defaults,
    and add time-based features.
    """
    lag_features = ['Temperature', 'Humidity', 'TDS Value', 'pH Level']
    lags = [1, 2, 3, 7]
    window = 7

    # Compute lag features
    for feature in lag_features:
        for lag in lags:
            df[f"{feature} Lag {lag}"] = df[feature].shift(lag)

    # Compute rolling statistics
    for feature in lag_features:
        df[f"{feature} Rolling Mean"] = df[feature].rolling(window=window).mean()
        df[f"{feature} Rolling Std"] = df[feature].rolling(window=window).std()

    # Time-based features
    df['Day of Week'] = df['Date'].dt.dayofweek + 1  # Monday=1 .. Sunday=7
    df['Month'] = df['Date'].dt.month

    # Fill missing values:
    for feature in lag_features:
        for lag in lags:
            col = f"{feature} Lag {lag}"
            df[col].fillna(df[feature], inplace=True)
        mean_col = f"{feature} Rolling Mean"
        std_col = f"{feature} Rolling Std"
        df[mean_col].fillna(df[feature], inplace=True)
        df[std_col].fillna(0, inplace=True)

    return df


@app.post("/predict")
def predict(window: List[SensorData]):
    try:
        # 1) Build DataFrame from incoming window
        records = []
        for r in window:
            records.append({
                'Date': datetime.strptime(r.date, "%Y-%m-%d"),
                'Temperature': r.temperature,
                'Humidity': r.humidity,
                'TDS Value': r.tds,
                'pH Level': r.ph
            })
        df = pd.DataFrame(records)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        # 2) Pad if fewer than 7 days
        if len(df) < 7:
            first_row = df.iloc[0]
            while len(df) < 7:
                new_row = first_row.copy()
                new_row['Date'] = new_row['Date'] - pd.Timedelta(days=1)
                df = pd.concat([pd.DataFrame([new_row]), df], ignore_index=True)
            df = df.sort_values('Date').reset_index(drop=True)

        # 3) Feature engineering
        feats = create_lagged_features(df)

        # 4) Select the last fully populated row
        last = feats.dropna().tail(1)
        if last.empty:
            raise HTTPException(status_code=400, detail="Insufficient data to compute features")

        # 5) Preprocess and predict
        X = preprocessor.transform(last)
        y = model.predict(X)
        return {"predicted_harvest_day": int(y[0])}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
