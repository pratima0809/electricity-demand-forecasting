# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
# pyrefly: ignore [missing-import]
import joblib
import json
# pyrefly: ignore [missing-import]
import plotly.express as px
from pathlib import Path
import pickle

# Set page config
st.set_page_config(page_title="Electricity Demand Forecast", page_icon="⚡", layout="wide")

# Paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "PJME_hourly.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "electricity_demand_model.pkl"
FEATURE_PATH = PROJECT_ROOT / "models" / "model_features.pkl"
METRICS_PATH = PROJECT_ROOT / "models" / "model_metrics.json"

# --- Styling ---
st.markdown("""
<style>
    .metric-container {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-value { font-size: 2rem; font-weight: bold; color: #3b82f6; }
    .metric-label { font-size: 1rem; color: #6b7280; font-weight: 500; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

# --- Load Data & Artifacts ---
@st.cache_data
def load_data():
    if not DATA_PATH.exists(): return None
    df = pd.read_csv(DATA_PATH)
    if 'Datetime' not in df.columns or 'PJME_MW' not in df.columns: return None
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    df.sort_index(inplace=True)
    return df

@st.cache_resource
def load_model_artifacts():
    model = None
    features = None
    metrics = None
    
    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
        except Exception as e:
            try:
                # Fallback to pickle
                with open(MODEL_PATH, 'rb') as f: model = pickle.load(f)
            except: pass

    if FEATURE_PATH.exists():
        try:
            with open(FEATURE_PATH, 'rb') as f: features = pickle.load(f)
        except:
            try: features = joblib.load(FEATURE_PATH)
            except: pass

    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, 'r') as f: metrics = json.load(f)
        except: pass
        
    return model, features, metrics

df = load_data()
model, features, metrics = load_model_artifacts()

# --- Main UI ---
st.title("Electricity Demand Forecasting Using Machine Learning")

# Error Handling for missing files
if df is None:
    st.error(f"Dataset not found at {DATA_PATH} or invalid columns. Please check the 'data' directory.")
    st.stop()
if model is None:
    st.error(f"Model not found at {MODEL_PATH} or failed to load. Please check the 'models' directory.")
    st.stop()
if features is None:
    st.error(f"Feature list not found at {FEATURE_PATH}. Please check the 'models' directory.")
    st.stop()
if metrics is None:
    st.warning(f"Metrics not found at {METRICS_PATH}. Model performance section will be limited.")

# 1. Project Overview
with st.expander("ℹ️ Project Overview", expanded=True):
    st.write("""
    This application predicts hourly electricity demand for the PJME region. 
    Accurate load forecasting is critical for efficient power grid operation, scheduling, and market bidding. 
    This dashboard provides exploratory data analysis, historical trends, and next-hour predictions using a pre-trained machine learning model.
    """)

# 2. Dataset Overview
st.sidebar.header("Dataset Overview")
st.sidebar.write(f"**Number of records:** {len(df):,}")
st.sidebar.write(f"**Date range:** {df.index.min().date()} to {df.index.max().date()}")
st.sidebar.write(f"**Minimum demand:** {df['PJME_MW'].min():,.0f} MW")
st.sidebar.write(f"**Maximum demand:** {df['PJME_MW'].max():,.0f} MW")
st.sidebar.write(f"**Average demand:** {df['PJME_MW'].mean():,.0f} MW")

# Feature Engineering function for on-the-fly EDA calculations
@st.cache_data
def get_eda_features(data):
    df_feat = data.copy()
    df_feat['Hour'] = df_feat.index.hour
    df_feat['DayOfWeek'] = df_feat.index.dayofweek
    df_feat['Month'] = df_feat.index.month
    df_feat['IsWeekend'] = df_feat['DayOfWeek'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
    return df_feat

df_eda = get_eda_features(df)

tab1, tab2, tab3 = st.tabs(["📊 Exploratory Analysis", "🔮 Next-Hour Forecast", "⚙️ Model Performance & Info"])

with tab1:
    st.header("Exploratory Analysis")
    
    st.subheader("Overall Electricity Demand Trend")
    resampled_df = df.resample('D').mean().reset_index()
    fig1 = px.line(resampled_df, x='Datetime', y='PJME_MW', title='Daily Average Electricity Demand', color_discrete_sequence=['#3b82f6'])
    st.plotly_chart(fig1, use_container_width=True)
    
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Average Demand by Hour")
        hourly_avg = df_eda.groupby('Hour')['PJME_MW'].mean().reset_index()
        fig2 = px.bar(hourly_avg, x='Hour', y='PJME_MW', color='PJME_MW', color_continuous_scale='Blues')
        st.plotly_chart(fig2, use_container_width=True)
        
        st.subheader("Weekday vs Weekend Demand")
        weekend_avg = df_eda.groupby('IsWeekend')['PJME_MW'].mean().reset_index()
        fig5 = px.bar(weekend_avg, x='IsWeekend', y='PJME_MW', color='IsWeekend', color_discrete_sequence=['#8b5cf6', '#10b981'])
        st.plotly_chart(fig5, use_container_width=True)
    
    with colB:
        st.subheader("Average Demand by Day of Week (0=Mon, 6=Sun)")
        weekly_avg = df_eda.groupby('DayOfWeek')['PJME_MW'].mean().reset_index()
        fig3 = px.bar(weekly_avg, x='DayOfWeek', y='PJME_MW', color='PJME_MW', color_continuous_scale='Greens')
        st.plotly_chart(fig3, use_container_width=True)
        
        st.subheader("Average Demand by Month")
        monthly_avg = df_eda.groupby('Month')['PJME_MW'].mean().reset_index()
        fig4 = px.bar(monthly_avg, x='Month', y='PJME_MW', color='PJME_MW', color_continuous_scale='Oranges')
        st.plotly_chart(fig4, use_container_width=True)

with tab2:
    st.header("Next-Hour Electricity Demand Forecast")
    
    if len(df) < 168:
        st.error("Not enough historical data to generate features. Minimum 168 hours required.")
    else:
        st.subheader("Recent Demand Trend")
        recent_data = df.tail(168).reset_index()
        fig_recent = px.line(recent_data, x='Datetime', y='PJME_MW', title='Actual Demand - Last 168 Hours', color_discrete_sequence=['#10b981'])
        st.plotly_chart(fig_recent, use_container_width=True)
        
        # Next-Hour Forecast Logic
        last_timestamp = df.index[-1]
        next_timestamp = last_timestamp + pd.Timedelta(hours=1)
        demand = df['PJME_MW']
        
        # Construct exact features manually for T+1 using ONLY historical data
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
        
        X_pred = pd.DataFrame([next_hour_dict])
        
        # Ensure all required features are present
        missing_features = [f for f in features if f not in X_pred.columns]
        if missing_features:
            st.error(f"Missing features for prediction: {missing_features}")
        else:
            X_pred = X_pred[features]
            
            # Validation to ensure no NaNs are passed to the model
            if X_pred.isna().any().any():
                st.error("Unable to generate forecast because required historical features contain missing values.")
            else:
                next_hour_pred = model.predict(X_pred)[0]
                
                st.markdown(f"""
                <div style="background-color: #eff6ff; border-left: 5px solid #3b82f6; padding: 20px; border-radius: 5px; margin-top: 20px; margin-bottom: 20px;">
                    <h3 style="color: #1e3a8a; margin-top: 0;">Next Hour Forecast</h3>
                    <h4 style="color: #3b82f6;">{next_timestamp.strftime('%d %b %Y, %H:%M')}</h4>
                    <h1 style="color: #2563eb; font-size: 3rem;">{next_hour_pred:,.0f} MW</h1>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("View Input Features used for this Forecast"):
                    st.dataframe(X_pred.T, use_container_width=True)

with tab3:
    st.header("Model Performance")
    
    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE", f"{metrics.get('MAE', 0):.2f}")
        c2.metric("RMSE", f"{metrics.get('RMSE', 0):.2f}")
        c3.metric("MAPE", f"{metrics.get('MAPE', 0):.2f}%")
        c4.metric("R²", f"{metrics.get('R2', 0):.4f}")
    else:
        st.info("Metrics not available.")
        
    st.markdown("---")
    st.header("Model Information")
    st.write("""
    This application uses the trained Gradient Boosting regression model and the engineered time/lag/rolling features to predict electricity demand.
    
    The model was trained on historical PJME hourly energy consumption data. To prevent data leakage and ensure realistic forecasting, lag features and rolling means are calculated exclusively from past observations.
    """)
    if metrics:
        st.write(f"**Model Type:** `{metrics.get('model_type', 'Unknown')}`")
        st.write(f"**Number of Features:** {metrics.get('num_features', 'Unknown')}")
