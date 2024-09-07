import numpy as np
import json
from .neural_network import NeuralNetwork
from .optimizers import SGD, Adam

def accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))

def save_params(neural_network: NeuralNetwork, filename='network_params.npz'):
    params = {}
    
    for i, layer in enumerate(neural_network.layers):
        if hasattr(layer, 'weights'):
            params[f'weights_layer_{i}'] = layer.weights
        if hasattr(layer, 'biases'):
            params[f'biases_layer_{i}'] = layer.biases
    
    if hasattr(neural_network, 'optimizer'):
        optimizer = neural_network.optimizer
        if isinstance(optimizer, Adam):
            params['adam_m'] = optimizer.m
            params['adam_v'] = optimizer.v
            params['adam_t'] = optimizer.t
        elif isinstance(optimizer, SGD):
            params['sgd_lr'] = optimizer.learning_rate

    np.savez(filename, **params)



def load_params(neural_network: NeuralNetwork, filename='network_params.npz'):
    data = np.load(filename)
    
    for i, layer in enumerate(neural_network.layers):
        if hasattr(layer, 'weights') and f'weights_layer_{i}' in data:
            layer.weights = data[f'weights_layer_{i}']
        if hasattr(layer, 'biases') and f'biases_layer_{i}' in data:
            layer.biases = data[f'biases_layer_{i}']
    
    if hasattr(neural_network, 'optimizer'):
        optimizer = neural_network.optimizer
        if isinstance(optimizer, Adam):
            optimizer.m = data['adam_m']
            optimizer.v = data['adam_v']
            optimizer.t = data['adam_t']
        elif isinstance(optimizer, SGD):
            optimizer.learning_rate = data['sgd_lr']


def save_results(results, filename='results.json'):
    with open(filename, 'w') as f:
        json.dump(results, f)

