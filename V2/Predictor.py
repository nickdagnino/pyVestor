import numpy as np
import pandas as pd

from datetime import datetime
import time

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Step 1: Data Collection
ticker = 'KO (3)'
data = pd.read_csv(f'Market Data\\{ticker}.csv')

#data['Date'] = pd.to_datetime(data['Date'])
#data['Date'] = data['Date'].apply(lambda x: time.mktime(x.timetuple()))
data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = (data['Date'] - data['Date'].min()).dt.days

#X = data[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].values.astype(np.float64)
X = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].values.astype(np.float64)
y = data['Close'].shift(-1).values.astype(np.float64)

X = X[:-1]
y = y[:-1]

# Step 2: Data Preprocessing
X = StandardScaler().fit_transform(X)

# Step 3: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, shuffle=True)

# Step 4: Initialize Parameters
weights = np.zeros(shape=(6,), dtype=np.float64)
bias = np.float64(0.0)

# Step 5: Model Training
max_iterations = 1000000
learning_rate = .015
tolerance = .0000001
gradient_clip_value = 1

cost_old = np.inf

for i in range(max_iterations):
    y_pred = np.dot(X_train, weights) + bias

    cost_new = np.mean((y_train-y_pred)**2)

    if i % 100 == 0:
        print(f"Iteration {i}: Cost = {cost_new}")

    if abs(cost_new - cost_old) < tolerance:
        print(f"Convereged at iteration {i}: Cost = {cost_new}")
        break

    weight_gradient = -2 * np.dot(X_train.T, (y_train-y_pred)) / len(X_train)
    bias_gradient = -2 * np.sum(y_train-y_pred) / len(X_train)

    weight_gradient = np.clip(weight_gradient, -gradient_clip_value, gradient_clip_value)
    bias_gradient = np.clip(bias_gradient, -gradient_clip_value, gradient_clip_value)

    weights -= learning_rate * weight_gradient
    bias -= learning_rate * bias_gradient

    cost_old = cost_new

# Step 6: Model Testing
y_pred_test = np.dot(X_test, weights) + bias
percent_diff = np.abs((y_test - y_pred_test) / y_test) * 100
accuracy = 100 - np.mean(percent_diff)

print(f"Model Accuracy: {accuracy}")
print(f"Weights: {weights}")
print(f"Bias: {bias}")

# Step 7: Model Prediction
X1 = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].values
X1 = StandardScaler().fit_transform(X1)

next_value = np.dot(X1[-1], weights) + bias

print(f"The predicted next value is: {next_value}")