# Apple Stock Analysis & Forecasting 📈

This repository provides a comprehensive technical analysis and price prediction of Apple Inc. (AAPL) stock using classical statistical methods, deep learning, and additive models.

## 🚀 Live Interactive Demo
**[Experience the AI-Powered Dashboard here]([https://nexttoken.app](https://apple-stock-analysis.nexttoken.app/))**
> *Note: This interactive dashboard was automatically generated from this repository's logic using NextToken's AI Agent.*

<img width="943" height="439" alt="Dashboard Overview" src="https://github.com/user-attachments/assets/c7f4f79a-ea78-4e89-b6f0-71bece784b48" />

---

## 📋 Project Overview
This project explores historical Apple stock data (1980–2024) to build and compare three distinct forecasting models:
1. **ARIMA**: Statistical approach for time-series stationarity and trend analysis.
2. **LSTM (Long Short-Term Memory)**: Deep learning architecture to capture long-term dependencies.
3. **Prophet**: Facebook’s additive model for handling seasonality and holiday effects.

## 📊 Model Evaluation Summary


| Model Type | Primary Metric | Result/Status | Key Strength |
| :--- | :--- | :--- | :--- |
| **ARIMA** | RMSE: 0.91 | $p=0.87$ (Ljung-Box) | Statistical Baseline & Trend |
| **LSTM** | Val Loss: 0.0004 | 31,901 Params | Complex Non-Linear Patterns |
| **Prophet** | Stan-Optimized | Converged | Seasonality & Long-term Growth |

---

## 🔍 Key Analysis Steps

### 1. Data Preprocessing & Stationarity
* Verified 11,094 data points with zero missing values.
* **ADF Test:** Confirmed non-stationarity ($p=1.0$), addressed via first-order differencing.

<img width="944" height="429" alt="Price Performance Chart" src="https://github.com/user-attachments/assets/34d2a06f-d1e0-48d1-99de-774a798e7171" />

### 2. Deep Learning Architecture (LSTM)
The model was built using a stacked LSTM approach with 31,901 trainable parameters, reaching convergence over 20 epochs with a validation loss of `4.2389e-04`.

<img width="942" height="437" alt="Forecasting Results" src="https://github.com/user-attachments/assets/b8b96d82-9212-405f-9c73-faf26eef523b" />

---

## 🛠️ Libraries & Tools
* **Data Handling:** `NumPy`, `Pandas`
* **Visualization:** `Seaborn`, `Matplotlib`
* **Machine Learning:** `Scikit-learn`, `TensorFlow/Keras`
* **Forecasting:** `Statsmodels` (ARIMA), `Prophet` (CmdStanPy)

## 🤝 Acknowledgments
Special thanks to **Alankar Jain** (NextToken) for providing the AI infrastructure to host the interactive version of this analysis.

## 📜 Conclusion
While ARIMA provides a strong statistical baseline with an RMSE of 0.91, the LSTM model excels at tracking non-linear swings, and Prophet offers the best interpretability regarding market seasonality.
