import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.serialize import model_from_json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

# --- Page Configuration ---
st.set_page_config(
    page_title="Delhi AQI Forecast Dashboard",
    page_icon="🌬️",
    layout="wide"
)

# --- AQI Calculation Function ---
def get_aqi_category(pm25):
    """Converts a PM2.5 value to an AQI category and color for plotting."""
    if pm25 <= 30: return "Good", "#4CAF50"
    elif pm25 <= 60: return "Satisfactory", "#8BC34A"
    elif pm25 <= 90: return "Moderate", "#FFEB3B"
    elif pm25 <= 120: return "Poor", "#FF9800"
    elif pm25 <= 250: return "Very Poor", "#F44336"
    else: return "Severe", "#B71C1C"

# --- Model and Data Loading (Cached for performance) ---
@st.cache_resource
def load_all_models():
    """Load all trained models from saved files."""
    # --- UPDATED: Load the new v2 Prophet model ---
    with open('prophet_model_v2.json', 'r') as fin:
        prophet_model = model_from_json(fin.read())
    
    # --- UPDATED: Load the new SARIMAX model ---
    sarimax_model = SARIMAXResults.load('sarimax_model.pkl')
    return prophet_model, sarimax_model

@st.cache_data
def load_and_prepare_data():
    """Load, combine, and prepare all necessary dataframes for the models."""
    df_original = pd.read_csv('data.csv', encoding='latin1')
    df_delhi = df_original[df_original['state'] == 'Delhi'].copy()
    df_processed = df_delhi[['date', 'pm2_5']].copy()
    df_processed['date'] = pd.to_datetime(df_processed['date'])
    df_processed.set_index('date', inplace=True)
    df_processed['pm2_5'].fillna(method='ffill', inplace=True)
    df_processed['pm2_5'].fillna(method='bfill', inplace=True)
    df_final = df_processed.loc[df_processed.index >= '2009-01-01'].copy()
    
    df_synthetic = pd.read_csv('data_2016_2024.csv', parse_dates=['date'], index_col='date')
    
    df_combined = pd.concat([df_final, df_synthetic])
    df_combined = df_combined[~df_combined.index.duplicated(keep='first')]
    df_combined.sort_index(inplace=True)
    df_combined.index = pd.to_datetime(df_combined.index)
    
    # Resample the DataFrame to keep its structure
    df_monthly = df_combined.resample('ME').mean().fillna(method='ffill')
    
    # Add event columns for SARIMAX
    df_monthly['stubble_burning'] = np.where((df_monthly.index.month == 10) | (df_monthly.index.month == 11), 1, 0)
    diwali_months = pd.to_datetime([
        '2009-10', '2010-11', '2011-10', '2012-11', '2013-11', '2014-10', '2015-11', 
        '2016-10', '2017-10', '2018-11', '2019-10', '2020-11', '2021-11', '2022-10', 
        '2023-11', '2024-11'
    ]).to_period('M')
    df_monthly['diwali'] = np.where(df_monthly.index.to_period('M').isin(diwali_months), 1, 0)
    
    return df_combined, df_monthly

# --- Load all assets ---
prophet_model, sarimax_model = load_all_models()
df_combined, df_monthly = load_and_prepare_data()

# --- App Header ---
st.title('Delhi Air Quality Forecast Dashboard 🌬️')
st.markdown("An interactive dashboard to forecast PM2.5 levels in Delhi and understand the key drivers of air pollution.")

# --- Define stubble burning function for Prophet ---
def stubble_burning_season(ds):
    return (pd.to_datetime(ds).month == 10 or pd.to_datetime(ds).month == 11)

# --- Key Metrics (KPIs) Section ---
st.header("Current State & Model Reliability")
col1, col2, col3 = st.columns(3)

last_recorded_pm25 = df_monthly['pm2_5'].iloc[-1]
last_aqi_cat, _ = get_aqi_category(last_recorded_pm25)
col1.metric("Last Recorded Monthly PM2.5 (Dec 2024)", f"{last_recorded_pm25:.2f}", help=f"This falls into the '{last_aqi_cat}' AQI category.")

# Create future event data for Jan 2025 forecast
future_exog_jan = pd.DataFrame(index=pd.date_range(start='2025-01-01', periods=1, freq='ME'))
future_exog_jan['stubble_burning'] = 0
future_exog_jan['diwali'] = 0
forecast_next_month = sarimax_model.forecast(steps=1, exog=future_exog_jan).iloc[0]
next_month_aqi_cat, _ = get_aqi_category(forecast_next_month)
col2.metric("Forecast for Next Month (Jan 2025)", f"{forecast_next_month:.2f}", help=f"The predicted AQI category is '{next_month_aqi_cat}'.")

monthly_test_data = df_monthly[df_monthly.index.year == 2024]
test_exog = monthly_test_data[['stubble_burning', 'diwali']]
sarimax_preds_2024 = sarimax_model.forecast(steps=len(monthly_test_data), exog=test_exog)
sarimax_rmse = np.sqrt(mean_squared_error(monthly_test_data['pm2_5'], sarimax_preds_2024))
col3.metric("SARIMAX Model Accuracy (RMSE)", f"{sarimax_rmse:.2f}", help="The average error of the model on 2024 data. Lower is better.")

st.divider()

# --- Main Forecast Section ---
st.header("The 2025 Forecast (SARIMAX Model)")
months_to_forecast = st.slider('Select number of months to forecast into 2025', 1, 12, 12)

# Create future event data for the full 2025 forecast
future_exog_2025 = pd.DataFrame(index=pd.date_range(start='2025-01-01', periods=12, freq='ME'))
future_exog_2025['stubble_burning'] = np.where((future_exog_2025.index.month == 10) | (future_exog_2025.index.month == 11), 1, 0)
diwali_2025 = pd.to_datetime(['2025-10']).to_period('M')
future_exog_2025['diwali'] = np.where(future_exog_2025.index.to_period('M').isin(diwali_2025), 1, 0)

forecast_2025 = sarimax_model.forecast(steps=months_to_forecast, exog=future_exog_2025.head(months_to_forecast))
forecast_df = pd.DataFrame({'Forecast': forecast_2025})
forecast_df['AQI Category'], forecast_df['Color'] = zip(*forecast_df['Forecast'].apply(get_aqi_category))

fig_sarima_final, ax_final = plt.subplots(figsize=(15, 7))
ax_final.plot(df_monthly['pm2_5'].loc[df_monthly.index.year >= 2023], label='Historical Monthly Average', marker='o', color='gray')
ax_final.bar(forecast_df.index, forecast_df['Forecast'], color=forecast_df['Color'], width=20, label='Predicted AQI Category')
ax_final.plot(forecast_df.index, forecast_df['Forecast'], linestyle='--', marker='o', color='red')
ax_final.set_title('Monthly PM2.5 Forecast for 2025 with AQI Categories', fontsize=16)
ax_final.set_ylabel('Monthly Average PM2.5')
ax_final.legend()
st.pyplot(fig_sarima_final)

with st.expander("View 2025 Forecast Data"):
    st.dataframe(forecast_df)

st.divider()

# --- Model Interpretation Section ---
st.header("Understanding the 'Why': Key Pollution Drivers")
st.markdown("The Prophet model helps us understand the specific events that cause pollution to spike. It has learned two major patterns from the historical data:")

future = prophet_model.make_future_dataframe(periods=365)
future['stubble_burning'] = future['ds'].apply(stubble_burning_season)
forecast = prophet_model.predict(future)
fig_components = prophet_model.plot_components(forecast)
st.pyplot(fig_components)

# --- Sidebar ---
st.sidebar.header("Model Performance")
st.sidebar.markdown("The following scores show the average error (RMSE) for each model when tested on 2024 data.")

daily_test_data = df_combined.loc[df_combined.index.year == 2024]['pm2_5']
prophet_preds_2024 = forecast[forecast['ds'].dt.year == 2024]['yhat'].values
prophet_rmse = np.sqrt(mean_squared_error(daily_test_data, prophet_preds_2024))

st.sidebar.metric("SARIMAX (Monthly)", f"{sarimax_rmse:.2f}")
st.sidebar.metric("Prophet (Daily)", f"{prophet_rmse:.2f}")