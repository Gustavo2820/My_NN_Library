import numpy as np
from .optimizers import SGD, Adam

class Dense:
    def __init__(self, input_dim, output_dim, activation=None, optimizer=SGD(learning_rate=0.01)):
        self.weights = np.random.randn(input_dim, output_dim) * 0.01 # Initialize weights
        self.biases = np.zeros((1, output_dim))  # Initialize biases
        self.activation = activation  # Activation function (if any)
        self.optimizer = optimizer

    def forward(self, inputs):
        self.inputs = inputs  # Store inputs
        self.z = np.dot(inputs, self.weights) + self.biases  # Compute weighted sum plus biases
        if self.activation:  # Apply activation function if provided
            self.a = self.activation.forward(self.z)
        else:
            self.a = self.z  # No activation
        return self.a
    
    def backward(self, dA):
        if self.activation:
            dZ = self.activation.backward(dA, self.z)
        else:
            dZ = dA
    
        self.dW = np.dot(self.inputs.T, dZ)  # Gradient w.r.t. weights
        self.dB = np.sum(dZ, axis=0, keepdims=True)  # Gradient w.r.t. biases
        dA_prev = np.dot(dZ, self.weights.T)  # Gradient w.r.t. previous layer inputs
        return dA_prev

    def update(self):
        self.optimizer.update(self)

class Dropout:
    def __init__(self, rate):
        self.rate = rate

    def forward(self, inputs, training=True):
        if not training:
            return inputs
        self.mask = np.random.binomial(1, 1 - self.rate, size=inputs.shape) / (1 - self.rate)
        return inputs * self.mask
    
    def backward(self, dA):
        return dA * self.mask
    
class BatchNormalization:
    def __init__(self, momentum=0.9, epsilon = 1e-8):
        self.momentum = momentum
        self.epsilon = epsilon

    def forward(self, inputs):
        self.mean = np.mean(inputs, axis=0)
        self.variance = np.var(inputs, axis=0)
        self.normalized_inputs = (inputs - self.mean) / np.sqrt(self.variance + self.epsilon)
        return self.normalized_inputs
    
    def backward(self, dA):
        N, D = dA.shape
        
        # Gradient w.r.t the variance
        dVar = np.sum(dA * (self.inputs - self.mean) * -0.5 * np.power(self.variance + self.epsilon, -1.5), axis=0)
        # Gradient w.r.t the mean
        dMean = np.sum(dA * -1 / np.sqrt(self.variance + self.epsilon), axis=0) + dVar * np.mean(-2 * (self.inputs - self.mean), axis=0)
        # Final Gradient
        dX = dA / np.sqrt(self.variance + self.epsilon) + dVar * 2 * (self.inputs - self.mean) / N + dMean / N
        return dX