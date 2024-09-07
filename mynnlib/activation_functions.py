import numpy as np

class ReLU:
    def forward(self, z):
        self.cache = z  # Store 'z' for use in the backward pass
        return np.maximum(0, z)  # Return max between 0 and 'z', element-wise
    
    def backward(self, dA, z):
        dZ = dA * (z > 0)  # Derivative of ReLU is 1 for positive 'z' and 0 otherwise
        return dZ  # Return the gradient to propagate it backward

class Sigmoid:
    def forward(self, z):
        self.cache = z  # Store 'z' for use in the backward pass
        return 1 / (1 + np.exp(-z))  # Sigmoid function
    
    def backward(self, dA, z):
        sigmoid = 1 / (1 + np.exp(-z))  # Calculate sigmoid(z) again
        dZ = dA * sigmoid * (1 - sigmoid)  # Derivative of sigmoid
        return dZ  # Return the gradient to propagate it backward

class Tanh:
    def forward(self, z):
        self.cache = z  # Store 'z' for use in the backward pass
        return np.tanh(z)  # Tanh function
    
    def backward(self, dA, z):
        tanh = np.tanh(z)  # Calculate tanh(z) again
        dZ = dA * (1 - tanh**2)  # Derivative of tanh
        return dZ  # Return the gradient to propagate it backward
