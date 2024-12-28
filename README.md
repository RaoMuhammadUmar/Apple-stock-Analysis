Documentation: Stock Market Price Prediction Using ARIMA, LSTM, and Prophet
1. Libraries and Tools
The following libraries were used for data processing, model building, and evaluation:
•	NumPy, Pandas: For data handling and manipulation.
•	Seaborn, Matplotlib: For visualizing the data and results.
•	MinMaxScaler (from Scikit-learn): Used to scale the stock price data before input into the LSTM model.
•	Keras: For building and training the LSTM model.
•	Statsmodels: For implementing ARIMA and conducting the ADF test to check stationarity.
•	Prophet: For time series forecasting, capturing seasonal trends in stock prices.
________________________________________
2. Data Loading and Visualization
The dataset used contains historical stock prices for Apple, loaded from a CSV file. The Date column is set as the index, and the stock features (Open, High, Low, Close) are analyzed.
•	Visualization: A line plot of the Close price is generated to visualize trends and fluctuations in the stock price over time.
Key Output/Insight:
•	The dataset consists of 11,094 data points from 1980 to 2024.
•	The plot shows the movement of the Apple stock price, highlighting long-term trends, spikes, and dips.
________________________________________
3. Data Preprocessing
The dataset is checked for its structure and the presence of missing values.
Key Output/Insight:
•	The dataset is complete with no missing values, as indicated by the .info() output.
•	The Close prices are the primary target for prediction.
________________________________________
4. Stationarity Check with ADF Test
The Augmented Dickey-Fuller (ADF) test is applied to the Close price series to determine its stationarity.
•	ADF Statistic: 4.8348
•	p-value: 1.0
Result: The p-value is greater than 0.05, indicating that the series is non-stationary. To address this, first-order differencing is applied to the data.
Key Output/Insight:
•	Since the series is non-stationary, a new column (Close_diff) is created with differenced values to help stabilize the mean of the series.
________________________________________
5. Correlation Heatmap
A heatmap is generated to visualize the correlation between the stock's Open, High, Low, and Close prices.
Key Output/Insight:
•	This heatmap shows the degree of correlation between the stock features. For example, the Close price is likely to have a strong correlation with High and Low, which can be useful for model feature selection.
________________________________________
6. ARIMA Model
The ARIMA model is applied to the Close price data to capture the time-dependent structure of the stock prices. The ARIMA model is configured with the parameters (2, 1, 2):
•	AR(2): Two lag observations in the autoregressive model.
•	I(1): First-order differencing to make the series stationary.
•	MA(2): Two lag observations in the moving average model.
Key Output/Insight:
•	The ARIMA model’s summary shows coefficients and significance values:
o	AR(1) coefficient: 0.6954, AR(2) coefficient: -0.4979
o	MA(1) coefficient: -0.7078, MA(2) coefficient: 0.4776
o	Model Fit: The model's AIC is 29,593.536 and BIC is 29,630.106, which are used for model comparison.
•	Residuals Analysis: The residuals of the ARIMA model are plotted and analyzed for randomness.
•	Ljung-Box Q-statistic: 0.03 (p-value: 0.87), which suggests no autocorrelation in the residuals.
Performance Metrics:
•	The MSE, RMSE, and MAE can be calculated to evaluate the model's performance. The ARIMA model’s performance will be assessed using these metrics based on the actual vs. predicted values.
________________________________________
7. LSTM Model
The LSTM model is built to capture long-term dependencies in the stock price data.
LSTM Architecture:
The LSTM model consists of:
1.	An LSTM layer with 50 units, followed by another LSTM layer with 50 units.
2.	A Dense layer with 25 units, followed by a final output layer predicting a single value (the stock price).
Model Summary:
•	Total Parameters: 31,901 trainable parameters.
•	Layers:
o	The LSTM layers learn from the time series data, capturing patterns and dependencies.
o	The Dense layers are used to make the final predictions.
Key Output/Insight:
•	The training loss gradually decreases with each epoch. For example:
o	Epoch 1: Loss = 7.8577e-05, Validation Loss = 0.0011
o	Epoch 20: Loss = 1.3718e-06, Validation Loss = 0.0018
•	The model demonstrates convergence over the training epochs.
Training Time: The model is trained for 20 epochs with a batch size of 32.
________________________________________
8. LSTM Predictions vs. True Prices
After training the LSTM model, predictions are made on the test set, and the results are compared to the true prices.
Key Output/Insight:
•	The predictions are plotted alongside the true stock prices, allowing for a visual comparison of the model’s performance.
•	LSTM Model Prediction: The LSTM model's predicted prices show how well it tracks the actual stock price movement over time.
________________________________________
9. Prophet Forecasting
The Prophet model is used for time series forecasting, incorporating yearly and weekly seasonality.
•	The model is fitted on the historical stock prices and then used to forecast future stock prices.
Key Output/Insight:
•	The Prophet model generates forecasts, which are plotted to show predicted trends in stock prices.
•	Components Plot: The individual seasonal components (e.g., yearly and weekly trends) are plotted to understand how the model captures underlying patterns.
________________________________________
10. Conclusion
In this analysis, multiple models (ARIMA, LSTM, Prophet) were applied to forecast Apple’s stock prices. Each model has its strengths:
•	ARIMA provides a statistical approach to modeling time series data with parameters tuned for the stock data's characteristics.
•	LSTM captures complex patterns and trends in the stock price through its deep learning architecture.
•	Prophet offers flexibility in handling seasonality and trends, making it suitable for time series forecasting.
The results from each model can be compared and evaluated using performance metrics such as MSE, RMSE, and MAE, as well as visualizations to determine the best-performing model for stock price prediction.

