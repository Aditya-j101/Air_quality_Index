import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Create a date range from 2016 to the end of 2024
dates = pd.date_range(start='2016-01-01', end='2024-12-31', freq='D')
df_synthetic = pd.DataFrame(dates, columns=['date'])
df_synthetic.set_index('date', inplace=True)

# 2. Create the base seasonal pattern (sine wave)
# We use a cosine wave so the peak is in winter (day 0/365)
days_in_year = 365.25
seasonal_component = 80 * -np.cos(2 * np.pi * (df_synthetic.index.dayofyear / days_in_year)) + 110

# 3. Create a slight downward long-term trend
total_days = len(df_synthetic)
trend_component = np.linspace(15, -15, total_days) # Starts at +15, ends at -15

# 4. Add realistic random noise (more noise in winter)
noise = np.random.normal(0, 15, total_days)
winter_months = (df_synthetic.index.month >= 10) | (df_synthetic.index.month <= 2)
winter_noise = np.random.normal(0, 40, total_days) # Extra noise for winter
noise[winter_months] += winter_noise[winter_months]

# 5. Combine components to create the initial PM2.5 value
df_synthetic['pm2_5'] = seasonal_component + trend_component + noise

# 6. Simulate the COVID-19 Lockdown Dip in 2020
lockdown_start = '2020-03-25'
lockdown_end = '2020-06-30'
df_synthetic.loc[lockdown_start:lockdown_end, 'pm2_5'] *= 0.6 # 40% reduction

# Ensure no negative pollution values
df_synthetic['pm2_5'] = df_synthetic['pm2_5'].clip(lower=10)

# --- Save to CSV ---
df_synthetic.to_csv('synthetic_data_2016_2024.csv')
print("Synthetic data saved to 'synthetic_data_2016_2024.csv'")

# --- Plotting the synthetic data for verification ---
plt.figure(figsize=(15, 6))
sns.lineplot(data=df_synthetic, x=df_synthetic.index, y='pm2_5')
plt.title('Generated Synthetic PM2.5 Data for Delhi (2016-2024)', fontsize=16)
plt.ylabel('PM2.5 Concentration')
plt.show()