import numpy as np
from .optimizers import SGD, Adam
from .utils import save_params, load_params

class Dense:
    def __init__(self, input_size, output_size, activation=None, optimizer=SGD(learning_rate=0.01)):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) * 0.01 # Initialize weights
        self.biases = np.zeros((1, output_size))  # Initialize biases
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

        # Verifique as dimensões
        print(f"inputs.T shape: {self.inputs.T.shape}")
        print(f"dZ shape: {dZ.shape}")

        # Calcule dW e verifique a forma
        self.dW = np.dot(self.inputs.T, dZ)  # (input_size, batch_size) dot (batch_size, output_size) -> (input_size, output_size)
        print(f"dW shape: {self.dW.shape}")
    
        # Calcule dB e verifique a forma
        self.dB = np.sum(dZ, axis=0, keepdims=True)  # (batch_size, output_size) -> (1, output_size)
        print(f"dB shape: {self.dB.shape}")
    
        # Calcule dA_prev e verifique a forma
        dA_prev = np.dot(dZ, self.weights.T)  # (batch_size, output_size) dot (output_size, input_size) -> (batch_size, input_size)
        print(f"dA_prev shape: {dA_prev.shape}")

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
    def __init__(self, momentum=0.9, epsilon=1e-8):
        self.momentum = momentum
        self.epsilon = epsilon
        self.gamma = np.ones(1)  # Initialize scale parameter
        self.beta = np.zeros(1)  # Initialize shift parameter
        self.running_mean = 0
        self.running_var = 0

    def forward(self, inputs, training=True):
        if training:
            # Calculate mean and variance
            self.mean = np.mean(inputs, axis=0)
            self.variance = np.var(inputs, axis=0)

            # Update running mean and variance
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.variance

            # Normalize inputs
            self.normalized_inputs = (inputs - self.mean) / np.sqrt(self.variance + self.epsilon)
            self.inputs = inputs

            # Scale and shift
            out = self.gamma * self.normalized_inputs + self.beta
            return out
        else:
            # During inference, use running mean and variance
            normalized_inputs = (inputs - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            return self.gamma * normalized_inputs + self.beta

    def backward(self, dA):
        N, D = dA.shape
        
        # Gradient with respect to gamma (scale parameter) and beta (shift parameter)
        dGamma = np.sum(dA * self.normalized_inputs, axis=0)
        dBeta = np.sum(dA, axis=0)

        # Gradient w.r.t. the normalized inputs
        dNormalized = dA * self.gamma
        
        # Gradient w.r.t. the variance
        dVar = np.sum(dNormalized * (self.inputs - self.mean) * -0.5 * np.power(self.variance + self.epsilon, -1.5), axis=0)
        # Gradient w.r.t. the mean
        dMean = np.sum(dNormalized * -1 / np.sqrt(self.variance + self.epsilon), axis=0) + dVar * np.mean(-2 * (self.inputs - self.mean), axis=0)
        # Final gradient w.r.t. inputs
        dX = dNormalized / np.sqrt(self.variance + self.epsilon) + dVar * 2 * (self.inputs - self.mean) / N + dMean / N
        
        return dX

    def update(self):
        # Update gamma and beta if they are being optimized
        if hasattr(self, 'optimizer'):
            self.gamma = self.optimizer.update(self.gamma, self.dGamma)
            self.beta = self.optimizer.update(self.beta, self.dBeta)
