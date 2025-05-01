def preprocess_data(lettuce_df):
    lag_features = ['Temperature', 'Humidity', 'TDS Value', 'pH Level']
    lags = [1, 2, 3, 7]

    for feature in lag_features:
        for lag in lags:
            lettuce_df[f"{feature} Lag {lag}"] = lettuce_df[feature].shift(lag)

    window = 7
    for feature in lag_features:
        lettuce_df[f"{feature} Rolling Mean"] = lettuce_df[feature].rolling(window=window).mean()
        lettuce_df[f"{feature} Rolling Std"] = lettuce_df[feature].rolling(window=window).std()

    # Ensure the index is datetime
    if not isinstance(lettuce_df.index, pd.DatetimeIndex):
        if 'Date' in lettuce_df.columns:
            lettuce_df['Date'] = pd.to_datetime(lettuce_df['Date'])
            lettuce_df.set_index('Date', inplace=True)
        else:
            raise ValueError("DataFrame must contain a 'Date' column to set as index.")

    lettuce_df['Day of Week'] = lettuce_df.index.dayofweek + 1
    lettuce_df['Month'] = lettuce_df.index.month

    return lettuce_df
