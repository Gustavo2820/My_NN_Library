import numpy as np

class NeuralNetwork:
    # Initialize the neural network
    def __init__(self, loss_func):
        self.layers = []  # List to store layers of the network
        self.loss = loss_func 

    def add_layer(self, layer):
        self.layers.append(layer)

    # Forward pass: Compute the output of the network
    # X: Input data
    def forward(self, X):
        self.activations = []  # List to store activations (a's) of each layer
        self.inputs = X  # Store input data for backpropagation
        for layer in self.layers:
            X = layer.forward(X)  # Forward pass through each layer
            self.activations.append(X)  # Store activations
        return X
    
    # Backward pass: Compute gradients for backpropagation
    # Y: True labels
    def backward(self, Y):
        # Compute gradient of the loss function with respect to the final layer's activation and the true labels
        dA = self.loss.backward(self.activations[-1], Y)
        # Backpropagate through each layer in reverse order
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    # Update parameters of the layers using gradient descent
    # learning_rate: Step size for parameter updates
    def update(self, learning_rate):
        for layer in self.layers:
            layer.update_params(learning_rate)
