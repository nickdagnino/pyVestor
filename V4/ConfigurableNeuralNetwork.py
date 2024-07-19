import numpy as np

# Class: NeuralNetwork
# Methods: __init__, initialize_weights_biases, sigmoid, sigmoid derivative, relu, relu_derivative, forward_pass, backpropagate, train, test
# About: An implementation of a neural network in which the user can manipukate both the number of hidden layers and the number of neuron in each
class NeuralNetwork:

    
    def __init__(self, input_size, hidden_layer_sizes, output_size):

        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size

        self.weights, self.biases = self.initialize_weights_biases()

    def initialize_weights_biases(self):

        weights = []
        biases = []

        layer_sizes = [self.input_size] + self.hidden_layer_sizes + [self.output_size]
    
        for i in range(1, len(layer_sizes)):

            weights.append(np.random.randn(layer_sizes[i-1], layer_sizes[i]) * np.sqrt(2.0 / layer_sizes[i-1]))
            biases.append(np.zeros((1, layer_sizes[i])))
    
        return weights, biases
    
    def sigmoid(self, x):

        return 1.0 / (1.0 + np.exp(-x))
    
    def sigmoid_derivative(self, x):

        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def relu(self, x):

        return np.maximum(0, x)

    def relu_derivative(self, x):

        return np.where(x > 0, 1, 0)

    def forward_pass(self, X, hidden_activation='relu', output_activation='linear'):
        
        activations = [X]

        for layer in range(len(self.weights)):

            z = np.dot(activations[-1], self.weights[layer]) + self.biases[layer]

            if layer < len(self.weights) - 1:

                a = self.relu(z) if hidden_activation == 'relu' else self.sigmoid(z)
            
            else:

                a = z if output_activation == 'linear' else self.sigmoid(z)
            
            activations.append(a)
        
        return activations[-1]
        
    def backpropagate(self, X, y, learning_rate=0.01, hidden_activation='relu', output_activation='linear'):

        activations = [X]
        zs = []  # Store intermediate weighted sums

        for layer in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[layer]) + self.biases[layer]
            zs.append(z)
            if layer < len(self.weights) - 1:
                a = self.relu(z) if hidden_activation == 'relu' else self.sigmoid(z)
            else:
                a = z if output_activation == 'linear' else self.sigmoid(z)
            activations.append(a)

        if output_activation == 'linear':
            delta = activations[-1] - y
        else:
            delta = (activations[-1] - y) * self.sigmoid_derivative(zs[-1])
    
        deltas = [delta]

        for layer in reversed(range(len(self.hidden_layer_sizes))):

            if hidden_activation == 'relu':

                delta_hidden = np.dot(deltas[-1], self.weights[layer+1].T) * self.relu_derivative(zs[layer])

            else:

                delta_hidden = np.dot(deltas[-1], self.weights[layer+1].T) * self.sigmoid_derivative(zs[layer])

            deltas.append(delta_hidden)

        deltas.reverse()

        for layer in range(len(self.weights)):

            actives = np.array(activations[layer].reshape(-1, 1))
            self.weights[layer] -= learning_rate * np.dot(actives, deltas[layer])
            self.biases[layer] -= learning_rate * np.sum(deltas[layer], axis=0, keepdims=True)

    def train(self, X, y, epochs=100, learning_rate=0.01, tolerance=1e-5, max_iterations_without_improvement=50):

        costs = []
        iterations_without_improvement = 0
        best_cost = float('inf')

        for epoch in range(epochs):

            total_loss = 0.0

            for i in range(len(X)):
                # Forward pass
                predictions = self.forward_pass(X[i])

                # Compute loss (mean squared error)
                loss = np.mean((predictions - y[i]) ** 2)
                total_loss += loss

                # Backpropagation
                self.backpropagate(X[i], y[i], learning_rate)

            avg_loss = total_loss / len(X)
            costs.append(avg_loss)

        # Check for improvement
            if avg_loss < best_cost - tolerance:
                best_cost = avg_loss
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1

            # Early stopping
            if iterations_without_improvement >= max_iterations_without_improvement:
                print(f"Early stopping at epoch {epoch+1}. No improvement for {max_iterations_without_improvement} iterations.")
                break

        # Optional: implement learning rate decay
        # learning_rate *= 0.99  # Reduce learning rate by 1% each epoch

            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss}")

        return costs  # Return cost history for plotting

    def test(self, X, y):
        predictions = self.forward_pass(X)
    
    # Mean Squared Error (MSE)
        mse = np.mean((y - predictions) ** 2)
    
    # Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mse)
    
    # Mean Absolute Error (MAE)
        mae = np.mean(np.abs(y - predictions))
    
    # R-squared (coefficient of determination)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - predictions) ** 2)
        r2 = 1 - (ss_residual / ss_total)
    
    # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((y - predictions) / y)) * 100
    
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"R-squared (R2) Score: {r2:}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape}%")
    
        return predictions
    
    def testV1(self, X, y):

        predictions = self.forward_pass(X)

        percent_diff = np.abs((y - predictions) / y) * 100
        accuracy = 100 - np.mean(percent_diff)

        print(f"Model Accuracy: {accuracy}")
        print(f"Weights: {self.weights}")
        print(f"Bias: {self.biases}")