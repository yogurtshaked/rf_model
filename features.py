# features.py
import pandas as pd
from datetime import timedelta

def create_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    lag_feats = ['Temperature', 'Humidity', 'TDS Value', 'pH Level']
    lags      = [1, 2, 3, 7]
    window    = 7

    for f in lag_feats:
        for lag in lags:
            df[f"{f} Lag {lag}"] = df[f].shift(lag)
        df[f"{f} Rolling Mean"] = df[f].rolling(window).mean()
        df[f"{f} Rolling Std"]  = df[f].rolling(window).std()

    df['Day of Week'] = df['Date'].dt.dayofweek + 1
    df['Month']       = df['Date'].dt.month

    # back-fill NaNs
    for f in lag_feats:
        for lag in lags:
            col = f"{f} Lag {lag}"
            df[col] = df[col].fillna(df[f])
        df[f"{f} Rolling Mean"] = df[f"{f} Rolling Mean"].fillna(df[f])
        df[f"{f} Rolling Std"]  = df[f"{f} Rolling Std"].fillna(0)

    return df


LAGS     = [1, 2, 3, 7]
MIN_ROWS = max(LAGS) + 1  # 8

def pad_to_minimum(df: pd.DataFrame) -> pd.DataFrame:
    """
    If df has fewer than MIN_ROWS rows, duplicate the first reading backward
    until df.length == MIN_ROWS. Then sort by Date and reset the index.
    This guarantees that when you later do .shift(7), the 8th row has
    a real value for lag-7 instead of NaN.
    """
    df = df.sort_values("Date").reset_index(drop=True)
    while len(df) < MIN_ROWS:
        first = df.iloc[0].copy()
        first["Date"] = first["Date"] - timedelta(days=1)
        df = pd.concat([pd.DataFrame([first]), df], ignore_index=True)
    return df.sort_values("Date").reset_index(drop=True)
