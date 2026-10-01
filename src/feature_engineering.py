import pandas as pd

def engineer_features(df):
    df = df.copy()
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['DayOfYear'] = df.index.dayofyear
    df['Year'] = df.index.year
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    df['Lag_1'] = df['PJME_MW'].shift(1)
    df['Lag_24'] = df['PJME_MW'].shift(24)
    df['Lag_168'] = df['PJME_MW'].shift(168)
    df['Rolling_Mean_24'] = df['PJME_MW'].rolling(window=24).mean()
    df['Rolling_Mean_168'] = df['PJME_MW'].rolling(window=168).mean()
    return df
