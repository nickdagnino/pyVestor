import numpy as np

class NeuralNetwork:

    def __init__ (self, input_size, hidden_layer_sizes, output_size):

        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size

        self.weights, self.biases = self.initialize_weights_biases()

    def initialize_weights_biases(self):
        
        weights = []
        biases = []

        weights.append(np.zeros(shape=(self.input_size, self.hidden_layer_sizes[0])))
        biases.append(np.zeros(shape=(1, self.hidden_layer_sizes[0])))

        for layer in range(1, len(self.hidden_layer_sizes)):

            weights.append(np.zeros(shape=(self.hidden_layer_sizes[layer-1], self.hidden_layer_sizes[layer])))
            biases.append(np.zeros(shape=(1, self.hidden_layer_sizes[layer])))

        weights.append(np.zeros(shape=(self.hidden_layer_sizes[-1], self.output_size)))
        biases.append(np.zeros(shape=(1, self.output_size)))

        return weights, biases
    
    def sigmoid(self, x):

        return 1.0 / (1.0 + np.exp(-x))
    
    def sigmoid_derivative(self, x):

        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def forward_pass(self, X):

        activations = [X]

        for layer in range(len(self.hidden_layer_sizes)):

            z = np.dot(activations[-1], self.weights[layer]) + self.biases[layer]
            a = self.sigmoid(z)
            activations.append(a)

            output_z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
            predictions = self.sigmoid(output_z)
            
            print(predictions)
            return predictions
        
    def backpropagate(self, X, y, learning_rate = 0.01):

        activations = [X]
        zs = []  # Store intermediate weighted sums

        for layer in range(len(self.hidden_layer_sizes)):
            z = np.dot(activations[-1], self.weights[layer]) + self.biases[layer]
            a = self.sigmoid(z)
            zs.append(z)
            activations.append(a)

        output_z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        predictions = self.sigmoid(output_z)

        delta_output = (predictions - y) * self.sigmoid_derivative(output_z)
        deltas = [delta_output]

        for layer in reversed(range(len(self.hidden_layer_sizes))):
            delta_hidden = np.dot(deltas[-1], self.weights[layer+1].T) * self.sigmoid_derivative(zs[layer])
            deltas.append(delta_hidden)

        for layer in range(len(self.hidden_layer_sizes)):
            actives = np.array(activations[layer].reshape(-1, 1))
            self.weights[layer] -= learning_rate * np.dot(actives, deltas[-layer-1])
            self.biases[layer] -= learning_rate * np.sum(deltas[-layer-1], axis=0)
            print(f"Weights: {self.weights} \n Biases: {self.biases}")

    def train(self, X, y, epochs=100, learning_rate= 0.01):

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
            #print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    def test(self, X, y):

        predictions = self.forward_pass(X)
        percent_diff = np.abs((y - predictions) / y) * 100
        accuracy = 100 - np.mean(percent_diff)

        print(f"Model Accuracy: {accuracy:.2f}%")
        print(f"Weights: {self.weights}")
        print(f"Bias: {self.biases[-1]}")