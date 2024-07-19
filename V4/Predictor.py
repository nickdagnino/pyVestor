import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from ConfigurableNeuralNetwork import NeuralNetwork

# Step 1: Data Collection
ticker = 'KO (3)'
data = pd.read_csv(f'Market Data\\{ticker}.csv')

X = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].values
y = data['Close'].shift(-1).values

X = X[:-1]
y = y[:-1]

# Step 2: Data Preprocessing
X = StandardScaler().fit_transform(X)
y = StandardScaler().fit_transform(y.reshape(-1, 1)).flatten()

# Step 3: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, shuffle=True)

# Step 4: Initalizing NeuralNetork as Predictor
Predictor = NeuralNetwork(6, [6, 6, 6], 1)

# Step 5: Model Training

#Predictor1.train(X_train, y_train, max_iterations, tolerance)
costs = Predictor.train(X_train, y_train, epochs=5000, learning_rate=0.0001)

# Optionally, plot the learning curve
plt.plot(costs)
plt.title('Training Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Step 6: Model Testing
predictions = Predictor.test(X_test, y_test)

Predictor.testV1(X_test, y_test)

# Optionally, visualize the predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
plt.show()