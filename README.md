


# Apple Stock Analysis & Forecasting 📈

This repository provides a comprehensive technical analysis and price prediction of Apple Inc. (AAPL) stock using classical statistical methods, deep learning, and additive models.

## 🚀 Live Interactive Demo
**[View the Live Dashboard Here](https://apple-stock-analysis.nexttoken.app/)**
> *Note: This interactive dashboard was automatically generated from this repository's logic using NextToken's AI Agent.*

---

## 📋 Project Overview
This project explores historical Apple stock data (1980–2024) to build and compare three distinct forecasting models:
1. **ARIMA**: Statistical approach for time-series stationarity and trend analysis.
2. **LSTM (Long Short-Term Memory)**: Deep learning architecture to capture long-term dependencies.
3. **Prophet**: Facebook’s additive model for handling seasonality and holiday effects.
<img width="942" height="437" alt="{7F7C4443-4B9E-4E81-A446-132ACAE3B20D}" src="https://github.com/user-attachments/assets/b8b96d82-9212-405f-9c73-faf26eef523b" />

## 🛠️ Libraries & Tools
* **Data Handling:** `NumPy`, `Pandas`
* **Visualization:** `Seaborn`, `Matplotlib`
* **Machine Learning:** `Scikit-learn` (MinMaxScaler)
* **Deep Learning:** `Keras` / `TensorFlow`
* **Forecasting:** `Statsmodels` (ARIMA), `Prophet`
<img width="943" height="439" alt="{9B5698B5-B67A-4DB8-B01A-CAFE6A1EC136}" src="https://github.com/user-attachments/assets/c7f4f79a-ea78-4e89-b6f0-71bece784b48" />
---

## 🔍 Key Analysis Steps

### 1. Data Preprocessing & Stationarity
* Verified 11,094 data points with zero missing values.
* **ADF Test (Augmented Dickey-Fuller):** Confirmed the series was non-stationary ($p=1.0$).
* **Differencing:** Applied first-order differencing to stabilize the mean for ARIMA modeling.

### 2. Model Performance
* **ARIMA (2, 1, 2):** Achieved an AIC of 29,593.5. Residual analysis via Ljung-Box test ($p=0.87$) confirmed no remaining autocorrelation.
* **LSTM:** A 4-layer architecture (50-50-25-1) trained over 20 epochs. Demonstrated strong convergence with minimal validation loss.
* **Prophet:** Successfully decomposed the stock's growth into yearly and weekly seasonal components.

## 📊 Visualizations
The project includes:
* **Correlation Heatmaps:** Analyzing relationships between Open, High, Low, and Close prices.
* **Residual Plots:** Diagnostic checks for model bias.
* **Prediction vs. Actual:** Comparative plots for LSTM and ARIMA accuracy.
<img width="944" height="429" alt="{D182F4EB-FBA4-48C9-994B-0DCBEA36F2FC}" src="https://github.com/user-attachments/assets/34d2a06f-d1e0-48d1-99de-774a798e7171" />

---

## 💻 Quick Start
To run the analysis locally:
1. Clone the repo: `git clone https://github.com`
2. Install dependencies: `pip install pandas numpy matplotlib seaborn statsmodels prophet tensorflow`
3. Open `apple stock market prediction.ipynb` in Jupyter or Colab.

## 🤝 Acknowledgments
Special thanks to **Alankar Jain** (NextToken) for reaching out and providing the AI infrastructure to host the interactive version of this analysis.

## 📜 Conclusion
While ARIMA provides a strong statistical baseline, the LSTM model excels at tracking non-linear movements, and Prophet offers the best interpretability regarding market seasonality.
