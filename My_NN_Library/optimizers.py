import numpy as np
from .neural_network import NeuralNetwork

# Stochastic Gradient Descent
class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, neural_network: NeuralNetwork):
        for layer in neural_network.layers:
            layer.weights -= self.learning_rate * layer.dW
            layer.biases -= self.learning_rate * layer.dB

# Adaptive Moment Estimation
class Adam:
    def __init__(self, learning_rate = 0.01, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = [] # Momentum
        self.v = [] # Variance
        self.t = 0 # Iterartion counter

    def init_m_v(self, layers):
        for layer in layers:
            self.m.append(np.zeros_like(layer.weights))
            self.v.append(np.zeros_like(layer.weights))

    def update(self, neural_network: NeuralNetwork):
        if not self.m:
            self.init_m_v(neural_network.layers)

        self.t += 1

        for i, layer in enumerate(neural_network.layers):
            self.m[i] = self.beta_1 * self.m[i] + (1 - self.beta_1) * layer.dW
            self.v[i] = self.beta_2 * self.v[i] + (1 - self.beta_2) * (layer.dW ** 2)
        
            m_hat = self.m[i]/(1 - self.beta_1)
            v_hat = self.v[i]/(1 - self.beta_2)

            layer.weights -= self.learning_rate * m_hat/(np.sqrt(v_hat) + self.epsilon)
            layer.biases -= self.learning_rate * m_hat/(np.sqrt(v_hat) + self.epsilon)