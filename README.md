# Electricity Demand Forecasting Using Machine Learning

## Problem Statement
Accurate electricity demand forecasting is crucial for utility companies to optimize power generation, ensure grid stability, and reduce operational costs. Imbalances between supply and demand can lead to energy waste or blackouts.

## Objective
To build and deploy a machine learning model capable of accurately forecasting next-hour electricity demand based on historical hourly data and engineered time-series features.

## Dataset
- **Source**: PJME Hourly Energy Consumption Dataset.
- **Target Variable**: `PJME_MW` (Megawatts).

## Features
- **Time Features**: Hour, DayOfWeek, Month, DayOfYear, Year, IsWeekend.
- **Lag Features**: Lag_1, Lag_24, Lag_168 (Historical past observations).
- **Rolling Features**: Rolling_Mean_24, Rolling_Mean_168.

## ML Workflow
1. Data ingestion and chronological sorting.
2. Feature engineering (time variables, lags, rolling means).
3. Model training and evaluation.
4. Deployment using Streamlit.

## Models Used
- **Gradient Boosting Regressor** (Scikit-Learn).

## Evaluation Metrics
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R²** (R-Squared)

## Streamlit Application Features
1. **Dataset Overview**: Interactive KPIs and historical bounds.
2. **Exploratory Analysis**: Daily trends, hourly profiles, and weekday vs. weekend breakdowns.
3. **Next-Hour Forecast**: Dynamic calculation of prediction features based exclusively on available historical data without target leakage.
4. **Model Performance**: Saved evaluation metrics (from hold-out test set).
5. **Model Information**: Details regarding the exact configurations and features used.

## Project Structure
```
electricity-demand-project/
├── .streamlit/
│   └── config.toml
├── data/
│   └── PJME_hourly.csv
├── models/
│   ├── electricity_demand_model.pkl
│   ├── model_features.pkl
│   └── model_metrics.json
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── evaluate_models.py
│   ├── feature_engineering.py
│   └── prediction.py
├── .gitignore
├── app.py
├── README.md
├── requirements.txt
└── test_pred.py
```

## How to Run Locally
1. Navigate to the project directory:
   ```bash
   cd electricity-demand-project
   ```
2. Create and activate a Python virtual environment (optional but recommended).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## How to Deploy on Streamlit Community Cloud
1. Push this directory to your GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and click "New app".
3. Point the application to this repository and select `app.py` as the main file path.
4. Click Deploy. Streamlit will automatically install the packages listed in `requirements.txt`.
