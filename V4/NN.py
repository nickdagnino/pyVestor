import numpy as np

class NN:

    def __init__(self, attributes, learning_rate):
        self.weights = np.zeros(shape=(attributes,))
        self.bias = 0.0
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return np.exp(-x) / (1 + 2*np.exp(-x) + np.exp(-2*x))
    
    def predict(self, input):
        layer_1 = np.dot(input, self.weights) + self.bias
        output_layer = self.sigmoid(layer_1)
        return output_layer
    
    def gradient_compute(self, input, target):
        layer_1 = np.dot(input, self.weights) + self.bias
        layer_2 = self.sigmoid(layer_1)
        prediction = layer_2

        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayer1 = self.sigmoid_derivative(layer_1)
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input)

        bias_gradient = (derror_dprediction * dprediction_dlayer1 * dlayer1_dbias)
        weight_gradient = (derror_dprediction * dprediction_dlayer1 * dlayer1_dweights)

        return weight_gradient, bias_gradient
    
    def parameter_update(self, weight_gradient, bias_gradient):
        self.weights -= self.learning_rate * weight_gradient
        self.bias -= self.learning_rate * bias_gradient
    
    def train(self, input, target, iterations, tolerance):
        
        cost_old = np.inf

        for iter in range(iterations):

            for element in range(len(input)):
                cost_new = np.mean((target[element]-self.predict(input[element]))**2)

                weight_gradient, bias_gradient = self.gradient_compute(input[element], target[element])
                self.parameter_update(weight_gradient, bias_gradient)

                #print(self.weights, self.bias)

            if iter % 100 == 0:
                print(f"Iteration {iter}: Cost = {cost_new}")

            if abs(cost_new - cost_old) < tolerance:
                print(f"Convereged at iteration {iter}: Cost = {cost_new}")
                break
            
            cost_old = cost_new

    def test(self, input, target):
        prediction = self.predict(input)
        percent_diff = np.abs((target - prediction) / target) * 100
        accuracy = 100 - np.mean(percent_diff)

        print(f"Model Accuracy: {accuracy}")
        print(f"Weights: {self.weights}")
        print(f"Bias: {self.bias}")

    