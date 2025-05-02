import pandas as pd
from datetime import timedelta

def pad_to_7(df: pd.DataFrame) -> pd.DataFrame:
    """
    If df has fewer than 7 rows, duplicate the first reading backward
    until you have exactly 7.
    """
    while len(df) < 7:
        first = df.iloc[0].copy()
        first['Date'] = first['Date'] - timedelta(days=1)
        df = pd.concat([pd.DataFrame([first]), df], ignore_index=True)
    return df.sort_values("Date").reset_index(drop=True)


def create_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute lags and rolls, but do NOT fill NaNs.
    """
    df = df.sort_values("Date").reset_index(drop=True)
    lag_feats = ['Temperature','Humidity','TDS Value','pH Level']
    lags      = [1,2,3,7]
    window    = 7

    # 1) raw shifts & rolling stats
    for f in lag_feats:
        for lag in lags:
            df[f"{f} Lag {lag}"] = df[f].shift(lag)
        df[f"{f} Rolling Mean"] = df[f].rolling(window).mean()
        df[f"{f} Rolling Std"]  = df[f].rolling(window).std()

    # 2) correct Excel-style Day of Week: Sunday=1 … Saturday=7
    df['Day of Week'] = ((df['Date'].dt.dayofweek + 1) % 7) + 1
    #    Mon=0→(1%7)+1=2, … Fri=4→(5%7)+1=6
    df['Month'] = df['Date'].dt.month

    return df
