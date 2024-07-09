import pandas as pd
import numpy as np

import time

ticker = 'KO (3)'
data = pd.read_csv(f'Market Data\\{ticker}.csv')

X = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].values
y = data['Close'].shift(-1).values

X = X[:-1]
y = y[:-1]

X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)

weights = np.zeros(shape=(6,))
bias = np.random.rand(1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward_propagation(X, weights, bias):
    return sigmoid(np.dot(X, weights) + bias)

def cost_function(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def back_propagation(X, y_true, y_pred, weights):
    d_weights = np.dot(X.T, (y_pred - y_true)) / len(y_true)
    d_biases = np.sum(y_pred - y_true) / len(y_true)
    return d_weights, d_biases

def update_parameters(weights, biases, d_weights, d_biases, learning_rate):
    weights -= learning_rate * d_weights
    biases -= learning_rate * d_biases
    return weights, biases

max_iterations = 10000
learning_rate = 0.01
tolerance = 0.01

old_cost = np.inf

for i in range(max_iterations):

    y_pred = forward_propagation(X_normalized, weights, bias)

    new_cost = cost_function(y, y_pred)

    if i % 100 == 0:
        print(f"Iteration {i}: Cost = {new_cost}")

    if abs(new_cost - old_cost) < tolerance:
        print(f"Convereged at iteration {i}: Cost = {new_cost}")
        break

    d_weights, d_biases = back_propagation(X_normalized, y, y_pred, weights)
    weights, bias = update_parameters(weights, bias, d_weights, d_biases, learning_rate)

    new_cost = old_cost





