import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Step 1: Data Collection
ticker = 'KO (3)'
data = pd.read_csv(f'Market Data\\{ticker}.csv')

X = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].values
y = data['Close'].shift(-1).values

X = X[:-1]
y = y[:-1]

# Step 2: Data Preprocessing
X = StandardScaler().fit_transform(X)

# Step 3: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, shuffle=True)

# Step 4: Initialize Parameters
weights = np.zeros(shape=(6,))
bias = 0.0

# Step 5: Model Training
max_iterations = 1000000
learning_rate = .01
tolerance = .0000000000001

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
next_value = np.dot(X[len(X)-1], weights) + bias

print(f"The predicted next value is: {next_value}")