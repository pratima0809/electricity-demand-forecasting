import pandas as pd
import joblib
from pathlib import Path

df = pd.read_csv('data/PJME_hourly.csv')
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)
df.sort_index(inplace=True)

model = joblib.load('models/electricity_demand_model.pkl')
import pickle
with open('models/model_features.pkl', 'rb') as f:
    features = pickle.load(f)

last_timestamp = df.index[-1]
next_timestamp = last_timestamp + pd.Timedelta(hours=1)
demand = df['PJME_MW']

lag_1 = demand.iloc[-1]
lag_24 = demand.iloc[-24]
lag_168 = demand.iloc[-168]

rolling_mean_24 = demand.iloc[-24:].mean()
rolling_mean_168 = demand.iloc[-168:].mean()

next_hour_dict = {
    "Hour": next_timestamp.hour,
    "DayOfWeek": next_timestamp.dayofweek,
    "Month": next_timestamp.month,
    "DayOfYear": next_timestamp.dayofyear,
    "Year": next_timestamp.year,
    "IsWeekend": int(next_timestamp.dayofweek >= 5),
    "Lag_1": lag_1,
    "Lag_24": lag_24,
    "Lag_168": lag_168,
    "Rolling_Mean_24": rolling_mean_24,
    "Rolling_Mean_168": rolling_mean_168
}

X_pred = pd.DataFrame([next_hour_dict])[features]
pred = model.predict(X_pred)[0]
print(f"Timestamp: {next_timestamp}")
print(f"Prediction: {pred}")
