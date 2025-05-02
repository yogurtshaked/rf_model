# features.py
import pandas as pd
from datetime import timedelta

LAGS      = [1, 2, 3, 7]
WINDOW    = 7
MIN_ROWS  = max(LAGS) + 1  # = 8

def pad_to_minimum(df: pd.DataFrame) -> pd.DataFrame:
    """
    If df has fewer than MIN_ROWS, 
    duplicate the first reading backward until
    you have exactly MIN_ROWS.
    """
    while len(df) < MIN_ROWS:
        first = df.iloc[0].copy()
        first['Date'] = first['Date'] - timedelta(days=1)
        df = pd.concat([pd.DataFrame([first]), df], ignore_index=True)
    return df.sort_values("Date").reset_index(drop=True)

def create_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").reset_index(drop=True)
    lag_feats = ['Temperature','Humidity','TDS Value','pH Level']

    # raw shifts & rolling
    for f in lag_feats:
        for lag in LAGS:
            df[f"{f} Lag {lag}"] = df[f].shift(lag)
        df[f"{f} Rolling Mean"] = df[f].rolling(WINDOW).mean()
        df[f"{f} Rolling Std"]  = df[f].rolling(WINDOW).std()

    # Excel-style Day of Week (Sunday=1…Saturday=7)
    df['Day of Week'] = ((df['Date'].dt.dayofweek + 1) % 7) + 1
    df['Month']       = df['Date'].dt.month

    return df
