import numpy as np

#Reference

# Input Layer Matrix [13733 x 6]
# Hidden Layer Matrix [6 x 1]
# Output Layer Matrix [1 x 1]

#I.H => 13733 x 1
#H.O => 13733 x 1 => Prediction for all input !!!! CORRECT


# Creating of NeuralNetwork Class
class NeuralNetwork:

    # Initialization
    def __init__(self, attributes, learning_rate):
        self.weights = np.zeros(shape=(1, attributes))
        self.bias = 0.0
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_deriv(self, x):
        #return self.sigmoid(x) * (1 - self.sigmoid(x))
        return np.dot(self.sigmoid(x), (1 - self.sigmoid(x)).T)
    
    def predict(self, input):
        #print(f"Input Shape from predict(): {input.shape}")
        #print(f'Weight Shape from predict(): {self.weights.shape}')
        #print(f'Weight Shape from predict() after .T: {self.weights.T.shape}')
        hidden_layer_1 = np.dot(input, self.weights.T) + self.bias
        #print(f'Hidden Layer Shape from predict(): {hidden_layer_1.shape}')
        output_layer = self.sigmoid(hidden_layer_1)
        print(f'Output Layer Shape from predict(): {output_layer.shape}')
        prediction = output_layer
        return prediction
    
    # Correct then validate the funtionality of compute_gradient() such that (1.) the computation of the gradient is correct
    def compute_gradients(self, input, target):
        #print(f'Weight Shape from compute_gradients(): {self.weights.T.shape}')
        hidden_layer_1 = np.dot(input, self.weights.T) + self.bias
        print(f'hidden_layer_1.shape: {hidden_layer_1.shape}')
        output_layer = self.sigmoid(hidden_layer_1)
        prediction = output_layer

        #print(f'prediction: {prediction.shape}, target: {np.array(target).reshape(-1, 1).shape}')
        derror_dprediction = 2 * (np.subtract(prediction, np.array(target).reshape(-1, 1))) # the derivative of error with respect to prediction
        dprediction_dlayer1 = [self.sigmoid_deriv(hidden_layer_1)]
        dlayer1_dbias = 1
        dlayer1_dweights = np.dot((1 * input), (0 * self.weights.T))

        derror_dbias = (derror_dprediction * dprediction_dlayer1 * dlayer1_dbias)
        #print(f'derror_dprediction: {derror_dprediction.shape}, dprediction_dlayer1: {dprediction_dlayer1}, dlayer1_dweights: {dlayer1_dweights.shape}')
        derror_dweights = (np.dot(np.dot(derror_dprediction, dprediction_dlayer1), dlayer1_dweights))

        return derror_dbias, derror_dweights

    # Correct then validate the funtionality of update_parameters() such that (1.) the computation of the gradient is correct
    def update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - derror_dbias * self.learning_rate
        #print(f'Weight Shape from update_parameters(): {self.weights.shape}')
        #print(f'derror_dweights Shape from update_parameters(): {derror_dweights.shape}')
        self.weights = self.weights - derror_dweights * self.learning_rate

    def trainBetter(self, inputs, targets, iterations, tolerance):
        
        cost_old = np.inf
        
        for i in range(iterations):
            #y_pred = np.dot(X_train, weights) + bias
            prediction = self.predict(inputs)

            cost_new = np.mean((targets-prediction)**2)
            print(cost_new)

            if i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost_new}")

            if abs(cost_new - cost_old) < tolerance:
                print(f"Convereged at iteration {i}: Cost = {cost_new}")
                break

            #weight_gradient = -2 * np.dot(X_train.T, (y_train-y_pred)) / len(X_train)
            #bias_gradient = -2 * np.sum(y_train-y_pred) / len(X_train)
            derror_dbias, derror_dweights = self.compute_gradients(inputs, targets)

            #weights -= learning_rate * weight_gradient
            #bias -= learning_rate * bias_gradient
            self.update_parameters(derror_dbias, derror_dweights)

            cost_old = cost_new

    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            # Pick a data instance at random
            random_data_index = np.random.randint(len(input_vectors))

            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]

            # Compute the gradients and update the weights
            derror_dbias, derror_dweights = self.compute_gradients(input_vector, target)

            self.update_parameters(derror_dbias, derror_dweights)

            # Measure the cumulative error for all the instances
            if current_iteration % 100 == 0:
                cumulative_error = 0
                # Loop through all the instances to measure the error
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]

                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)

                    cumulative_error = cumulative_error + error
                cumulative_errors.append(cumulative_error)

        return cumulative_errors