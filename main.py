# server.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from supabase import create_client, Client
from datetime import datetime

# ————— Supabase client (use service‐role key here, NOT anon key) —————
SUPABASE_URL = "https://cudfnejuropkvoifwzao.supabase.co"
SUPABASE_SERVICE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImN1ZGZuZWp1cm9wa3ZvaWZ3emFvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDU5MTM4MDQsImV4cCI6MjA2MTQ4OTgwNH0.riIgH4d2eeAImiL622Esnk2Ub1e0HM1scJfyvRgTFAs"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ————— Load your preprocessor & model —————
preprocessor = joblib.load('preprocessor.pkl')
model        = joblib.load('model.pkl')

app = FastAPI()

class BatchRequest(BaseModel):
    batch_id: str

def create_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    lag_features = ['Temperature','Humidity','TDS Value','pH Level']
    lags = [1,2,3,7]
    window = 7

    for feature in lag_features:
        for lag in lags:
            df[f"{feature} Lag {lag}"] = df[feature].shift(lag)
        df[f"{feature} Rolling Mean"] = df[feature].rolling(window).mean()
        df[f"{feature} Rolling Std"]  = df[feature].rolling(window).std()

    df['Day of Week'] = df['Date'].dt.dayofweek + 1
    df['Month']       = df['Date'].dt.month

    return df

@app.post('/predict')
def predict(req: BatchRequest):
    # 1) fetch last 7 days for this batch
    resp = supabase\
      .table('readings')\
      .select('date,temperature,humidity,tds,ph')\
      .eq('batch_id', req.batch_id)\
      .order('date', desc=True)\
      .limit(7)\
      .execute()

    rows = resp.data or []
    if len(rows) < 7:
        return {"error": "Need at least 7 days of readings."}

    # 2) build DataFrame, sort oldest→newest
    df = pd.DataFrame(rows)
    df['Date'] = pd.to_datetime(df['date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # 3) compute lags & rolls, then take the final fully-populated row
    feat_df = create_lagged_features(df)
    last_row = feat_df.dropna().tail(1)

    # 4) transform & predict
    X = preprocessor.transform(last_row)
    y = model.predict(X)

    return {"predicted_harvest_day": int(y[0])}
