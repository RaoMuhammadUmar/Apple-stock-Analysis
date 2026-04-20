# Apple Stock Analysis & Forecasting 📈
> Comparing Statistical (ARIMA), Deep Learning (LSTM), and Additive (Prophet) Models

This repository provides a comprehensive end-to-end pipeline for analyzing and forecasting Apple Inc. (AAPL) stock prices. It moves from raw data processing to a live, production-ready dashboard.

## 🚀 Live Interactive Demo
**[Experience the AI-Powered Dashboard here](https://nexttoken.app)**
> *Note: This dashboard was automatically generated from this repository's logic using NextToken's AI Agent.*

<img width="943" height="439" alt="Dashboard Overview" src="https://github.com" />

---

## 📋 Project Objective
The goal of this project is to evaluate the predictive power of three distinct methodologies on high-volatility tech stock data (1980–2024):
*   **Statistical Baseline:** Using ARIMA to model linear relationships and stationarity.
*   **Sequential Modeling:** Using LSTMs to capture long-term temporal dependencies that traditional stats miss.
*   **Decomposition:** Using Prophet to isolate yearly, weekly, and daily seasonal effects.

---

## 📊 Model Evaluation Summary


| Model Type | Primary Metric | Validation Status | Key Strength |
| :--- | :--- | :--- | :--- |
| **ARIMA (2,1,2)** | **RMSE: 0.91** | $p=0.87$ (Ljung-Box) | Excellent for short-term trend stability. |
| **LSTM (RNN)** | **Val Loss: 0.0004** | 20 Epochs | Captures complex, non-linear price swings. |
| **Prophet** | **Stan-Optimized** | Converged | Handles seasonality and holiday growth. |

---

## 🔍 Technical Deep-Dive

### 1. Data Preprocessing & Stationarity (The ADF Test)
Raw stock data is rarely stationary. To prepare the data for the ARIMA model, I performed an **Augmented Dickey-Fuller (ADF) test**.
*   **Result:** Initial p-value was $1.0$, indicating a non-stationary unit root.
*   **Solution:** Applied **First-Order Differencing** to stabilize the mean and variance, making the data suitable for forecasting.

<img width="944" height="429" alt="Price Performance Chart" src="https://github.com" />

### 2. LSTM Architecture (Neural Network Design)
To handle the sequential nature of stock prices, I designed a **Stacked LSTM** architecture:
*   **Layer 1 & 2:** 50 LSTM units each to extract high-level temporal features.
*   **Dropout/Dense Layers:** 25 hidden units to prevent overfitting.
*   **Trainable Parameters:** 31,901 parameters trained using the Adam optimizer.
*   **Performance:** Achieved convergence with a final loss of `1.21e-06`, tracking actual price movements with high fidelity.

<img width="942" height="437" alt="Forecasting Results" src="https://github.com" />

### 3. Prophet Seasonality Decomposition
Prophet was utilized to decompose the stock's growth into:
*   **Trend:** The overall 40-year trajectory of Apple.
*   **Seasonality:** Identifying specific months or days where Apple historically performs better (e.g., product launch cycles).

---

## 🛠️ Tech Stack
*   **Core:** `Python 3.x`, `Jupyter Notebook`
*   **ML/DL:** `TensorFlow/Keras`, `Scikit-learn`
*   **Time Series:** `Statsmodels` (SARIMAX), `Prophet`
*   **Deployment:** `NextToken AI Agent`

## 🤝 Acknowledgments
Special thanks to **Alankar Jain** (ex-Google) for the **NextToken** infrastructure, which allowed for the rapid deployment of this analysis into a functional web application.

## 📜 Final Conclusion
The analysis shows that while **LSTM** provides the best "visual" fit for non-linear swings, the **ARIMA** model remains remarkably accurate for 1-step ahead forecasting ($RMSE=0.91$). For long-term strategic insights, **Prophet** provides the most interpretable seasonal trends.
