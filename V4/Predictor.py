import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from NN import NN
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

# Step 3: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, shuffle=True)

# Step 4: Initalizing NeuralNetork as Predictor

num_attributes = 6
learning_rate = 0.01
max_iterations = 10000
tolerance = .000000001

#Predictor1 = NN(num_attributes, learning_rate)
Predictor2 = NeuralNetwork(6, [6, 6], 1)

# Step 5: Model Training

#Predictor1.train(X_train, y_train, max_iterations, tolerance)
Predictor2.train(X_train, y_train)

# Step 6: Model Testing

Predictor2.test(X_test, y_test)