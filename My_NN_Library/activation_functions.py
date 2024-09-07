import numpy as np

class ReLU:
    # Forward pass: Applies ReLU activation function
    def forward(self, z):
        self.cache = z  # Store 'z' for use in the backward pass
        return np.maximum(0, z)  # Return max between 0 and 'z', element-wise
    
    # Backward pass: Compute gradient of ReLU function with respect to 'z'
    def backward(self, dA, z):
        dZ = dA * (z > 0)  # Derivative of ReLU is 1 for positive 'z' and 0 otherwise
        return dZ  # Return the gradient to propagate it backward

class Sigmoid:
    # Forward pass: Applies the sigmoid activation function
    def forward(self, z):
        self.cache = z  # Store 'z' for use in the backward pass
        return 1 / (1 + np.exp(-z))  # Sigmoid function
    
    # Backward pass: Compute the gradient of the sigmoid function with respect to 'z'
    def backward(self, dA, z):
        sigmoid = 1 / (1 + np.exp(-z))  # Calculate sigmoid(z) again
        dZ = dA * sigmoid * (1 - sigmoid)  # Derivative of sigmoid
        return dZ  # Return the gradient to propagate it backward
