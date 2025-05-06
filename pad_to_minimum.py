# features.py
import pandas as pd
from datetime import timedelta

# if your lags are [1,2,3,7], you need at least max(lags)+1 = 8 rows
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
