# features.py
import pandas as pd
from datetime import timedelta

def create_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with columns
      ['Date','Temperature','Humidity','TDS Value','pH Level']
    this will:
      • sort by Date
      • compute lags 1,2,3,7
      • compute 7‐day rolling mean & std
      • extract Day of Week / Month
      • fill any NaNs exactly as you did before
    """
    df = df.sort_values("Date").reset_index(drop=True)
    lag_feats = ['Temperature','Humidity','TDS Value','pH Level']
    lags      = [1,2,3,7]
    window    = 7

    # 1) raw shifts & rolls  
    for f in lag_feats:
        for lag in lags:
            df[f"{f} Lag {lag}"] = df[f].shift(lag)
        df[f"{f} Rolling Mean"] = df[f].rolling(window).mean()
        df[f"{f} Rolling Std"]  = df[f].rolling(window).std()

    # 2) time features
    df['Day of Week'] = df['Date'].dt.dayofweek + 1
    df['Month']       = df['Date'].dt.month

    # 3) fill‐in same‐as‐today defaults for any NaNs
    for f in lag_feats:
        for lag in lags:
            df[f"{f} Lag {lag}"].fillna(df[f], inplace=True)
        df[f"{f} Rolling Mean"].fillna(df[f], inplace=True)
        df[f"{f} Rolling Std"].fillna(0, inplace=True)

    return df


def pad_to_7(df: pd.DataFrame) -> pd.DataFrame:
    """
    If df has <7 rows, duplicate the first row backward until it has exactly 7.
    """
    while len(df) < 7:
        first = df.iloc[0].copy()
        first['Date'] = first['Date'] - timedelta(days=1)
        df = pd.concat([pd.DataFrame([first]), df], ignore_index=True)
    return df.sort_values("Date").reset_index(drop=True) from fastapi 
