import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "PJME_hourly.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "electricity_demand_model.pkl"
FEATURE_PATH = PROJECT_ROOT / "models" / "model_features.pkl"
METRICS_PATH = PROJECT_ROOT / "models" / "model_metrics.json"

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    df.sort_index(inplace=True)
    
    print("Loading model and features...")
    import joblib
    with open(MODEL_PATH, 'rb') as f:
        model = joblib.load(f)
        
    print(f"Model type: {type(model)}")
        
    with open(FEATURE_PATH, 'rb') as f:
        features = pickle.load(f)
        
    print(f"Features: {features}")
    
    # Feature engineering
    print("Engineering features...")
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
    
    df.dropna(inplace=True)
    
    # Evaluate
    print("Evaluating model...")
    X = df[features]
    y_true = df['PJME_MW']
    
    y_pred = model.predict(X)
    
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        "model_type": str(type(model)),
        "num_features": len(features),
        "features": features,
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "R2": float(r2)
    }
    
    print("Metrics calculated:", metrics)
    
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Metrics saved to {METRICS_PATH}")

if __name__ == "__main__":
    main()
