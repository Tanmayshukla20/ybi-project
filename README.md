import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Load stock data (example: Apple Inc.)
ticker = "AAPL"
df = yf.download(ticker, start="2018-01-01", end="2024-12-31")

# Step 2: Use only the 'Close' price
df = df[['Close']].dropna()

# Step 3: Create a column for prediction
forecast_days = 30
df['Prediction'] = df[['Close']].shift(-forecast_days)

# Step 4: Prepare the features (X) and labels (y)
X = np.array(df.drop(['Prediction'], axis=1))[:-forecast_days]
y = np.array(df['Prediction'])[:-forecast_days]

# Step 5: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")

# Step 8: Predict the next N days
X_future = df.drop(['Prediction'], axis=1)[-forecast_days:]
forecast = model.predict(X_future)

# Step 9: Plot the results
plt.figure(figsize=(12, 6))
plt.title(f"{ticker} Stock Price Prediction")
plt.xlabel("Days")
plt.ylabel("Price (USD)")
plt.plot(df['Close'], label='Actual Price')
plt.plot(df.index[-forecast_days:], forecast, label='Predicted Price', linestyle='--')
plt.legend()
plt.grid()
plt.show()
