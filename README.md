# Delhi Air Quality (AQI) Forecasting Dashboard 🌬️

An end-to-end time series forecasting project that predicts future PM2.5 air pollution levels in Delhi. The project culminates in an interactive web application built with Streamlit that provides forecasts, explains key pollution drivers, and compares the performance of various advanced forecasting models.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aditya-j101-air-quality-index-app-zsy75f.streamlit.app/)  ---

## 📊 Live Dashboard & Key Features

The final models and analysis are deployed as an interactive Streamlit dashboard designed for intuitive interpretation.



* **KPI Dashboard:** High-level metrics showing the last recorded pollution level, the forecast for next month, and the primary model's accuracy.
* **Intelligent Forecasting:** The dashboard uses advanced models (SARIMAX and Prophet) that understand not only the base seasonal patterns but also the impact of special, real-world events.
* **Model Interpretability:** Explains the "why" behind the forecast by visualizing the specific pollution impact of **Diwali** and the annual **stubble burning season**.
* **Performance Comparison:** Compares the accuracy (RMSE) of the different models to provide a clear view of their reliability.

---

## 📝 Project Approach & Methodology

This project followed a standard end-to-end machine learning workflow, with a strong emphasis on data relevance and model intelligence.

### 1. The Data Gap: Original vs. Modern Data
The project began with a historical dataset (`data.csv`) that only contained data up to 2015. A crucial early insight was that a model trained on this outdated data would produce an irrelevant forecast for the present day.

### 2. Solution: Research-Driven Synthetic Data Augmentation
To bridge this gap, a realistic **synthetic dataset** was generated for the years 2016-2024. This was not random data; it was carefully engineered based on extensive research into Delhi's recent pollution patterns to mimic key real-world phenomena:
* The significant drop in pollution during the **2020 COVID-19 lockdown**.
* The continuation of the strong yearly seasonality (winter spikes, monsoon dips).
* A slight overall downward trend in annual average pollution.

> **⚠️ Important Caveat:** While this synthetic data makes the model far more relevant than using the old data alone, it is still a simulation. The model's performance on this combined dataset should be interpreted as a strong estimate, but it is not a substitute for a complete, real-world dataset.

### 3. Advanced Modeling with Domain Knowledge
With a complete and relevant dataset, two advanced time-series models were trained and compared:

* **SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables):** This powerful classical model was trained on monthly data. It was enhanced by adding external regressors ("dummy variables") to explicitly inform the model about the occurrence of **Diwali** and the **stubble burning season**. This model emerged as the most accurate in terms of RMSE and is used for the final 2025 forecast.

* **Prophet:** Facebook's open-source forecasting library was used for its excellent ability to model seasonality and incorporate the effects of special events. An enhanced version was trained to specifically learn and visualize the impact of **Diwali** (as a holiday) and the **stubble burning season** (as a custom seasonality).

### 4. Deployment
The final, intelligent models were deployed into an interactive web application using **Streamlit**, making the complex analysis and forecasts accessible and understandable to a general audience.

---

## 💡 Key Insights & Project Learnings

* **Data Relevance is Paramount:** The most significant learning was that a model is only as good as its data. Using outdated information will produce an irrelevant forecast. Data augmentation through research-backed synthesis proved to be a critical solution.
* **Domain Knowledge Transforms Models:** A simple model can only repeat past patterns. By explicitly teaching the models about real-world events like Diwali and farm fires, we created much more intelligent and realistic forecasting tools.
* **SARIMAX for Accuracy, Prophet for Insight:** While SARIMAX provided the most accurate monthly forecast (lowest RMSE), the Prophet model was invaluable for interpreting and visualizing *why* these seasonal spikes occur, making it a powerful tool for explaining the results.

---

## 🛠️ Tech Stack

* **Data Analysis & Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib
* **Machine Learning & Forecasting:**
    * **Prophet:** For daily forecasting and modeling special events.
    * **Statsmodels:** For training the advanced SARIMAX model.
    * **Scikit-learn:** For calculating performance metrics (RMSE).
* **Development Environment:** Jupyter Notebook
* **Web Application & Deployment:** Streamlit

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
