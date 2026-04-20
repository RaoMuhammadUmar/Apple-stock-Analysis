# Apple Stock Analysis & Multi-Model Forecasting
> **Benchmarking ARIMA, LSTM, and Prophet for Financial Time-Series Excellence**

This repository contains a professional-grade data science pipeline designed to analyze over 40 years of Apple Inc. (AAPL) historical data (1980–2024). The project evaluates the effectiveness of traditional econometrics against modern deep learning and Bayesian trend decomposition.

## Live Interactive Experience
To demonstrate the production-readiness of this logic, the entire analysis has been deployed into a live, interactive AI dashboard.

**[View the Live Dashboard on NextToken](https://nexttoken.app)**
> *Note: This interactive dashboard was automatically generated from this repository's logic using NextToken's AI Agent.*

<img width="943" height="439" alt="Dashboard Overview" src="https://github.com" />

---

## Project Overview
Predicting stock prices is a high-variance problem. This project implements three distinct analytical lenses to solve it:
1.  **Classical Econometrics:** Using **ARIMA** to handle linear trends and auto-regressive properties.
2.  **Sequential Deep Learning:** Implementing **Long Short-Term Memory (LSTM)** networks to capture hidden non-linear patterns.
3.  **Additive Trend Modeling:** Leveraging **Facebook Prophet** for robust handling of seasonality and long-term growth trajectories.

---

## Model Evaluation Summary


| Model Type | Primary Metric | Validation Status | Core Strength |
| :--- | :--- | :--- | :--- |
| **ARIMA (2,1,2)** | **RMSE: 0.91** | p=0.87 (Ljung-Box) | Statistical Rigor & Trend |
| **LSTM (RNN)** | **Val Loss: 0.0004** | 20 Epochs / 31k Params | Non-Linear Volatility |
| **Prophet** | **Stan-Optimized** | Fully Converged | Seasonal Market Shifts |

---

## Technical Deep-Dive

### 1. Data Preprocessing & Stationarity (The ADF Test)
Financial data is non-stationary by nature. To prepare the data for the ARIMA model, I performed an **Augmented Dickey-Fuller (ADF) test**.
*   **Discovery:** Initial p-value was 1.0, indicating a non-stationary unit root.
*   **Optimization:** Applied **First-Order Differencing** to stabilize the mean and variance.

<img width="944" height="429" alt="Price Performance Chart" src="https://github.com" />

**ARIMA Statistical Output:**
```text
Model: ARIMA(2, 1, 2)   Log Likelihood: -14791.768
AIC: 29593.536          Ljung-Box (Q): 0.03 (p=0.87)
------------------------------------------------------
ar.L1: 0.6954 (p=0.00)  ma.L1: -0.7078 (p=0.00)
ar.L2: -0.4979 (p=0.00) ma.L2: 0.4776 (p=0.00)


2. Deep Learning Architecture (Stacked LSTM)
To handle sequential dependencies, I architected a stacked LSTM network with 31,901 trainable parameters.
Model Architecture:
Layer 1 & 2: 50 LSTM units each to extract temporal features across time steps.
Layer 3 & 4: Dense (25 units) and Output (1 unit) for final price prediction.
Training Log Snapshot:

Epoch 1/20  - loss: 4.77e-05 - val_loss: 0.0011
Epoch 10/20 - loss: 1.52e-06 - val_loss: 4.03e-04
Epoch 20/20 - loss: 1.21e-06 - val_loss: 4.23e-04

3. Prophet Seasonality Decomposition
Prophet was utilized to decompose the stock's growth into macro-trends and seasonal cycles, capturing the underlying rhythm of Apple's price action, including yearly product cycles and holiday surges.
Tooling & Tech Stack
Languages: Python 3.x, Jupyter Notebook
ML/DL: TensorFlow, Keras, Scikit-learn
Time Series: Statsmodels (SARIMAX), Prophet (CmdStanPy)
UI/Deployment: NextToken AI Agent
Acknowledgments
Special thanks to Alankar Jain (ex-Google) for providing the NextToken infrastructure, which allowed for the rapid deployment of this analysis into a production-grade web application.
Final Conclusion
The analysis shows that while LSTM provides the best visual fit for non-linear swings, the ARIMA model remains remarkably accurate for 1-step ahead forecasting (RMSE=0.91). For long-term strategic insights, Prophet provides the most interpretable seasonal trends.
