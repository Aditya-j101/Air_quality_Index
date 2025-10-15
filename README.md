# Delhi Air Quality (AQI) Forecasting Dashboard 🌬️

An end-to-end time series forecasting project that predicts future PM2.5 air pollution levels in Delhi. The project culminates in an interactive web application built with Streamlit that provides forecasts, explains key pollution drivers, and compares the performance of various models.

 Link to App
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aditya-j101-air-quality-index-app-puulo1.streamlit.app/)  ---

## 📊 Live Dashboard

The final model and analysis are deployed as an interactive Streamlit dashboard.



**Key Features:**
* **KPI Dashboard:** High-level metrics showing the last recorded pollution level, the forecast for next month, and the primary model's accuracy.
* **Interactive Forecasting:** A slider to select the number of months to forecast into 2025 using the best-performing model (SARIMA).
* **Intuitive Visualizations:** Forecasts are displayed with color-coded Air Quality Index (AQI) categories (Good, Moderate, Severe, etc.) for easy interpretation.
* **Model Interpretability:** Explains the "why" behind the forecast by visualizing the impact of key events like Diwali and the farm fire season, as learned by the Prophet model.
* **Performance Comparison:** Compares the accuracy (RMSE) of different models tested during the project.

---

## 📂 Project Structure

```
delhi-aqi-forecast/
│
├── models/                # Saved model files
│   ├── prophet_model_v2.json
│   └── sarima_model.pkl
│
├── data/                  # Datasets used
│   ├── data.csv
│   └── synthetic_data_2016_2024.csv
│
├── app.py                 # The main Streamlit application script
├── project.ipynb          # Jupyter Notebook with all analysis and modeling
└── requirements.txt       # Python libraries needed to run the app
```

---

## 📝 Project Methodology

This project followed a standard end-to-end machine learning workflow:

### 1. Problem Definition
The primary goal was to forecast future daily and monthly PM2.5 concentrations for Delhi. The success of the models was quantitatively measured using the **Root Mean Squared Error (RMSE)**.

### 2. Data Sourcing and Augmentation
* The initial dataset (`data.csv`) contained historical air quality data from 1990 to 2015.
* **Crucial Insight:** This data was too outdated to make a relevant forecast for the present day.
* **Solution:** A realistic **synthetic dataset** (`synthetic_data_2016_2024.csv`) was generated based on extensive research into Delhi's recent pollution patterns. This new data accurately mimics key real-world events, including:
    * The 2020 COVID-19 lockdown pollution dip.
    * The consistent yearly seasonality (winter spikes, monsoon dips).
    * A slight overall downward trend in annual average pollution.

### 3. Modeling and Iteration
Several models were trained and evaluated to find the best fit for the data:

* **Prophet:** Used for its excellent ability to model seasonality and incorporate the effects of special events. An enhanced version was trained to specifically learn the impact of **Diwali** and the **stubble burning season**.
* **SARIMA (Seasonal AutoRegressive Integrated Moving Average):** A powerful classical statistical model. After resampling the data to a monthly frequency, **SARIMA proved to be the most accurate model** in terms of RMSE. It was chosen as the primary model for the final 2025 forecast.
* **ETS (Exponential Smoothing):** Another strong classical model that was evaluated but was slightly outperformed by SARIMA.
* **XGBoost:** Explored initially but removed to simplify the final app and focus on the best-performing time-series-specific models.

### 4. Deployment
The final application was built using **Streamlit** and deployed to the cloud, making the models' insights accessible through a user-friendly web interface.

---

## 💡 Key Insights

* **Seasonality is King:** The most dominant driver of Delhi's air pollution is the strong yearly cycle, with severe pollution consistently occurring in the winter months.
* **Special Events Matter:** Explicitly modeling real-world events like Diwali (firecrackers) and the October-November farm fire season significantly improves the nuance and accuracy of the forecast.
* **SARIMA for Stability, Prophet for Insight:** While SARIMA provided the most accurate monthly forecast, the Prophet model was invaluable for interpreting and visualizing *why* these seasonal spikes occur.

---

## 🛠️ Technologies Used

* **Python:** The core programming language.
* **Pandas & NumPy:** For data manipulation and analysis.
* **Matplotlib & Seaborn:** For data visualization.
* **Statsmodels:** For training the SARIMA and ETS models.
* **Prophet:** For forecasting with special event handling.
* **Scikit-learn:** For calculating performance metrics (RMSE).
* **Streamlit:** For building and deploying the interactive web dashboard.
* **Jupyter Notebook:** For exploratory analysis and model development.

---

## 🚀 How to Run Locally

To run this application on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[YOUR-USERNAME]/[YOUR-REPO-NAME].git
    cd [YOUR-REPO-NAME]
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```
    * On Windows: `venv\Scripts\activate`
    * On macOS/Linux: `source venv/bin/activate`

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

The application will open in your web browser.

