# 📈 Stock Market Price Prediction Using ARIMA, LSTM, and Prophet

## 📚 Documentation

---

## 🛠️ Libraries and Tools

The following libraries were used for data processing, model building, and evaluation:

* **NumPy, Pandas** – Data handling and manipulation
* **Seaborn, Matplotlib** – Data visualization
* **MinMaxScaler (Scikit-learn)** – Feature scaling for LSTM
* **Keras** – Building and training the LSTM model
* **Statsmodels** – ARIMA modeling and ADF test for stationarity
* **Prophet** – Time series forecasting with seasonality

---

## 📊 Data Loading and Visualization

The dataset contains historical stock prices for Apple, loaded from a CSV file.

* The `Date` column is set as the index
* Features analyzed: **Open, High, Low, Close**

### 🔍 Key Insights

* Dataset contains **11,094 data points (1980–2024)**
* Line plot of **Close price** shows long-term trends, spikes, and dips

---

## 🧹 Data Preprocessing

* Dataset structure examined
* Missing values checked

### 🔍 Key Insights

* No missing values found (`.info()` confirms completeness)
* **Close price** selected as the prediction target

---

## 📉 Stationarity Check (ADF Test)

* **ADF Statistic:** 4.8348
* **p-value:** 1.0

### 🔍 Result

* Series is **non-stationary** (p > 0.05)
* Applied **first-order differencing**

### 🔍 Key Insight

* New column `Close_diff` created to stabilize the series

---

## 🔗 Correlation Heatmap

* Visualizes relationships between **Open, High, Low, Close**

### 🔍 Key Insight

* Strong correlation between:

  * Close ↔ High
  * Close ↔ Low

---

## 📈 ARIMA Model

Model configuration: **ARIMA(2,1,2)**

* **AR(2):** 2 lag observations
* **I(1):** First-order differencing
* **MA(2):** 2 lag observations

### 🔍 Key Insights

* AR(1): 0.6954

* AR(2): -0.4979

* MA(1): -0.7078

* MA(2): 0.4776

* **AIC:** 29,593.536

* **BIC:** 29,630.106

### 📊 Residual Analysis

* Ljung-Box Q-statistic: **0.03 (p = 0.87)**
* Indicates **no autocorrelation** → good model fit

### 📏 Evaluation Metrics

* MSE
* RMSE
* MAE

---

## 🤖 LSTM Model

Designed to capture long-term dependencies in time series data.

### 🧠 Architecture

* LSTM Layer (50 units)

* LSTM Layer (50 units)

* Dense Layer (25 units)

* Output Layer (1 unit)

* **Total Parameters:** 31,901

### 🔍 Key Insights

* Training loss decreases over epochs:

  * Epoch 1: Loss = 7.8577e-05, Val Loss = 0.0011
  * Epoch 20: Loss = 1.3718e-06, Val Loss = 0.0018

* Model shows **good convergence**

* Training: **20 epochs, batch size 32**

---

## 📊 LSTM Predictions vs Actual

* Predictions plotted against true prices

### 🔍 Key Insight

* Model effectively tracks stock price trends over time

---

## 🔮 Prophet Forecasting

* Handles **trend + seasonality (weekly & yearly)**
* Forecasts future stock prices

### 🔍 Key Insights

* Generates future predictions with confidence intervals
* Components plot shows:

  * Trend
  * Weekly patterns
  * Yearly seasonality

---

## 🏁 Conclusion

This project applied three models for stock prediction:

### 📌 ARIMA

* Statistical approach
* Effective for structured time series

### 📌 LSTM
<img width="584" height="314" alt="{A7AF4896-944C-4CF9-B1BF-664F04076C33}" src="https://github.com/user-attachments/assets/6f6485e5-3946-4574-ab4b-1a75cd4f8d75" />


* Deep learning model
* Captures complex, non-linear patterns

### 📌 Prophet
<img width="668" height="345" alt="{76B47CD6-9841-4BC0-9750-9CFD762F3ADE}" src="https://github.com/user-attachments/assets/4216de19-1f55-46bf-94fb-974ed6992760" />

* Handles seasonality and trend effectively
* Easy to interpret and visualize

### 📊 Final Evaluation

Models can be compared using:

* **MSE**
* **RMSE**
* **MAE**
* Visual performance plots

➡️ The best model depends on prediction accuracy and use case.

---
