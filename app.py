import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.serialize import model_from_json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

# Page Configuration 
st.set_page_config(
    page_title="Delhi AQI Forecast Dashboard",
    page_icon="🌬️",
    layout="wide"
)

# AQI Calculation Function 
def get_aqi_category(pm25):
    """Converts a PM2.5 value to an AQI category and color."""
    if pm25 <= 30:
        return "Good", "#4CAF50"  # Green
    elif pm25 <= 60:
        return "Satisfactory", "#8BC34A"  # Light Green
    elif pm25 <= 90:
        return "Moderate", "#FFEB3B"  # Yellow
    elif pm25 <= 120:
        return "Poor", "#FF9800"  # Orange
    elif pm25 <= 250:
        return "Very Poor", "#F44336"  # Red
    else:
        return "Severe", "#B71C1C"  # Dark Red

# Model and Data Loading
@st.cache_resource
def load_all_models():
    with open('prophet_model.json', 'r') as fin:
        prophet_model = model_from_json(fin.read())
    sarima_model = SARIMAXResults.load('sarima_model.pkl')
    return prophet_model, sarima_model

@st.cache_data
def load_and_prepare_data():
    df_original = pd.read_csv('data.csv', encoding='latin1')
    df_delhi = df_original[df_original['state'] == 'Delhi'].copy()
    df_processed = df_delhi[['date', 'pm2_5']].copy()
    df_processed['date'] = pd.to_datetime(df_processed['date'])
    df_processed.set_index('date', inplace=True)
    df_processed.sort_index(inplace=True)
    df_processed['pm2_5'].fillna(method='ffill', inplace=True)
    df_processed['pm2_5'].fillna(method='bfill', inplace=True)
    df_final = df_processed.loc[df_processed.index >= '2009-01-01'].copy()
    
    df_synthetic = pd.read_csv('data_2016_2024.csv', parse_dates=['date'], index_col='date')
    
    df_combined = pd.concat([df_final, df_synthetic])
    df_combined = df_combined[~df_combined.index.duplicated(keep='first')]
    df_combined.sort_index(inplace=True)
    df_combined.index = pd.to_datetime(df_combined.index)
    
    df_monthly = df_combined['pm2_5'].resample('ME').mean().fillna(method='ffill')
    
    return df_combined, df_monthly

# Load all assets 
prophet_model, sarima_model = load_all_models()
df_combined, df_monthly = load_and_prepare_data()

#  Main App Body 
st.title('Delhi Air Quality (PM2.5) Forecast Dashboard 🌬️')
st.markdown("This dashboard presents the 2025 air quality forecast using the best-performing model, **SARIMA**, and compares its performance against other models.")

# NEW: KPI Cards Section 
st.header("Key Metrics")
col1, col2, col3 = st.columns(3)

# KPI 1: Last Recorded PM2.5
last_recorded_pm25 = df_monthly.iloc[-1]
last_recorded_aqi, last_recorded_color = get_aqi_category(last_recorded_pm25)
col1.metric("Last Recorded Monthly PM2.5 (Dec 2024)", f"{last_recorded_pm25:.2f}", help=f"AQI Category: {last_recorded_aqi}")

# KPI 2: Forecast for Next Month
forecast_next_month = sarima_model.forecast(steps=1).iloc[0]
next_month_aqi, next_month_color = get_aqi_category(forecast_next_month)
col2.metric("Forecast for Next Month (Jan 2025)", f"{forecast_next_month:.2f}", help=f"Predicted AQI Category: {next_month_aqi}")

# KPI 3: SARIMA Model Accuracy
monthly_test_data = df_monthly[df_monthly.index.year == 2024]
sarima_preds_2024 = sarima_model.forecast(steps=len(monthly_test_data))
sarima_rmse = np.sqrt(mean_squared_error(monthly_test_data, sarima_preds_2024))
col3.metric("SARIMA Model RMSE on 2024 Data", f"{sarima_rmse:.2f}", help="The lower the Root Mean Squared Error, the better the model's accuracy.")

# Final SARIMA Forecast with AQI Colors 
st.header("1. Final 2025 Forecast (SARIMA Model)")
months_to_forecast = st.slider(
    label='Select number of months to forecast into 2025',
    min_value=1, max_value=12, value=12, step=1
)

forecast_2025 = sarima_model.forecast(steps=months_to_forecast)
forecast_df = pd.DataFrame({'Forecast': forecast_2025})
forecast_df['AQI Category'], forecast_df['Color'] = zip(*forecast_df['Forecast'].apply(get_aqi_category))

# Plot the final SARIMA forecast
fig_sarima_final, ax_final = plt.subplots(figsize=(15, 7))
ax_final.plot(df_monthly.loc[df_monthly.index.year >= 2023], label='Historical Monthly Average', marker='o', color='gray')
# Add colored bars for AQI category 
ax_final.bar(forecast_df.index, forecast_df['Forecast'], color=forecast_df['Color'], width=20, label='AQI Category')
ax_final.plot(forecast_df.index, forecast_df['Forecast'], label=f'2025 Forecast', linestyle='--', marker='o', color='red')
ax_final.set_title('Final SARIMA Forecast for 2025 with AQI Categories', fontsize=16)
ax_final.set_ylabel('Monthly Average PM2.5')
ax_final.legend()
ax_final.grid(True)
st.pyplot(fig_sarima_final)

# NEW: Display Forecast Data in a Table 
with st.expander("View 2025 Forecast Data"):
    st.dataframe(forecast_df)

# Performance Comparison Section 
st.header("2. Prophet Model Performance (on 2024 Data)")

daily_test_data = df_combined.loc[df_combined.index.year == 2024]['pm2_5']
prophet_future = prophet_model.make_future_dataframe(periods=365)
prophet_all_preds = prophet_model.predict(prophet_future)
prophet_preds_2024 = prophet_all_preds[prophet_all_preds['ds'].dt.year == 2024]['yhat'].values
prophet_rmse = np.sqrt(mean_squared_error(daily_test_data, prophet_preds_2024))

fig_p, ax_p = plt.subplots()
ax_p.plot(daily_test_data.index, daily_test_data, label='Actuals')
ax_p.plot(daily_test_data.index, prophet_preds_2024, label='Prophet', linestyle='--')
ax_p.set_title(f'Prophet Performance (RMSE: {prophet_rmse:.2f})')
ax_p.legend()
st.pyplot(fig_p)

# Sidebar 
st.sidebar.header("Model RMSE Scores (2024)")
st.sidebar.markdown(f"**SARIMA (Monthly):** `{sarima_rmse:.2f}`")
st.sidebar.markdown(f"**Prophet (Daily):** `{prophet_rmse:.2f}`")