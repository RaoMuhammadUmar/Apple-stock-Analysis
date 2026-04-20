# 📈 Apple Stock Analysis & Forecasting (1980–2024)

A complete **end-to-end time series forecasting project** that analyzes and predicts **Apple Inc. (AAPL)** stock prices using:

- 📊 Classical Statistical Modeling (**ARIMA**)  
- 🤖 Deep Learning (**LSTM Neural Networks**)  
- 📅 Additive Time-Series Modeling (**Prophet**)  
- 🌐 Interactive Web Dashboard (AI-generated deployment)

---

## 🚀 Live Interactive Dashboard  
👉 **[Explore the AI-Powered Dashboard](https://apple-stock-analysis.nexttoken.app/)**  

> This dashboard was automatically generated using NextToken’s AI agent, transforming the notebook logic into a fully interactive experience.

![Dashboard Overview](https://github.com/user-attachments/assets/c7f4f79a-ea78-4e89-b6f0-71bece784b48)

---

## 📋 Project Overview  

This project performs a **comprehensive financial time-series analysis** on Apple stock data spanning **1980 to 2024 (11,094 records)**.

### Workflow:
- Data loading and visualization  
- Stationarity testing and transformation  
- Statistical modeling (ARIMA)  
- Deep learning forecasting (LSTM)  
- Additive modeling (Prophet)  
- Model evaluation and comparison  
- Deployment via interactive dashboard  

---

## 📊 Model Comparison Summary  

| Model | Performance | Strength |
|------|------------|---------|
| **ARIMA (2,1,2)** | RMSE: **0.91** | Strong statistical baseline |
| **LSTM** | Val Loss: **0.0004** | Captures nonlinear patterns |
| **Prophet** | Converged | Best for seasonality interpretation |

---

## 🔍 Methodology & Pipeline  

### 📌 1. Data Loading & Visualization  
- Dataset: Apple historical stock prices  
- Features: Open, High, Low, Close  
- No missing values (clean dataset)

![Price Performance Chart](https://github.com/user-attachments/assets/34d2a06f-d1e0-48d1-99de-774a798e7171)

---

### 📌 2. Stationarity Testing  
- **ADF Test Result:**  
  - Statistic: `4.83`  
  - p-value: `1.0` → ❌ Non-stationary  

✔ Applied **first-order differencing**

---

### 📌 3. Statistical Modeling (ARIMA)  
- Model: **ARIMA(2,1,2)**  
- Ljung-Box p-value: **0.87** (good fit)

**Performance Metrics:**
- MSE: `0.8427`  
- RMSE: `0.9180`  
- MAE: `0.2967`  

✔ Acts as a **baseline forecasting model**

---

### 📌 4. Deep Learning Model (LSTM)  

**Architecture:**
- 2 × LSTM layers (50 units each)  
- Dense layers (25 → 1)  
- Total Parameters: **31,901**

**Training:**
- Epochs: 20  
- Batch Size: 32  
- Final Validation Loss: **4.2389e-04**

![Forecasting Results](https://github.com/user-attachments/assets/b8b96d82-9212-405f-9c73-faf26eef523b)

✔ Best at capturing **complex nonlinear price movements**

---

### 📌 5. Prophet Forecasting  

- Handles **trend + seasonality + long-term growth**
- Built on **Stan probabilistic framework**
- Detects:
  - Yearly trends  
  - Weekly patterns  

✔ Most **interpretable model**

---

## 📉 Visual Analysis Included  

- Price trend over time  
- ACF / PACF plots  
- Correlation heatmap  
- Residual distribution  
- Forecast comparison  
- Training vs validation loss  
- LSTM predicted vs actual prices  

---

## 🛠️ Tech Stack  

### 📊 Data & Visualization  
- `NumPy`, `Pandas`  
- `Matplotlib`, `Seaborn`  

### 🤖 Machine Learning  
- `Scikit-learn`  
- `TensorFlow / Keras`  

### 📈 Time-Series Modeling  
- `Statsmodels` (ARIMA)  
- `Prophet`  

### 🌐 Deployment  
- NextToken AI Dashboard  

---

## 📂 Project Structure  
apple-stock-analysis/
│
├── data/
│ └── AAPL_historical_data.csv
│
├── notebooks/
│ └── analysis.ipynb
│
├── models/
│ ├── arima_model.pkl
│ └── lstm_model.h5
│
├── dashboard/
│ └── deployed_app_link.txt
│
├── images/
│ └── (plots and visual outputs)
│
├── requirements.txt
└── README.md


---

## ⚙️ How to Run Locally  

```bash
# Clone repo
git clone https://github.com/your-username/apple-stock-analysis.git

# Navigate into project
cd apple-stock-analysis

# Install dependencies
pip install -r requirements.txt

# Run Jupyter Notebook
jupyter notebook

🎯 Key Insights
Stock data is non-stationary, requiring transformation
ARIMA works well for linear trends
LSTM significantly improves performance for nonlinear patterns
Prophet provides business-friendly interpretability
🤝 Acknowledgments
Dataset: Kaggle (Apple Historical Data)
Dashboard Hosting: Alankar Jain (NextToken)
📜 Conclusion

This project demonstrates how multiple forecasting techniques complement each other:

📊 ARIMA → Reliable statistical baseline
🤖 LSTM → Best predictive performance
📅 Prophet → Best interpretability

Together, they provide a robust hybrid approach to financial forecasting.
