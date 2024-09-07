import numpy as np
import os
import json

def accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))

def save_params(neural_network, filename='network_params.npz'):
    params = {}
    
    for i, layer in enumerate(neural_network.layers):
        if hasattr(layer, 'weights'):
            params[f'weights_layer_{i}'] = layer.weights
        if hasattr(layer, 'biases'):
            params[f'biases_layer_{i}'] = layer.biases
    
    if hasattr(neural_network, 'optimizer'):
        from .optimizers import SGD, Adam
        optimizer = neural_network.optimizer
        if isinstance(optimizer, Adam):
            params['adam_m'] = optimizer.m
            params['adam_v'] = optimizer.v
            params['adam_t'] = optimizer.t
        elif isinstance(optimizer, SGD):
            params['sgd_lr'] = optimizer.learning_rate

    np.savez(filename, **params)

def load_params(neural_network, filename='network_params.npz'):
    if not os.path.isfile(filename) or os.path.getsize(filename) == 0:
        for i, layer in enumerate(neural_network.layers):
            if hasattr(layer, 'weights'):
                layer.weights = np.random.randn(layer.input_size, layer.output_size) * 0.01
            if hasattr(layer, 'biases'):
                layer.biases = np.zeros(layer.output_size)
        
        if hasattr(neural_network, 'optimizer'):
            from .optimizers import SGD, Adam
            optimizer = neural_network.optimizer
            if isinstance(optimizer, Adam):
                optimizer.m = np.zeros_like(neural_network.layers[0].weights)
                optimizer.v = np.zeros_like(neural_network.layers[0].weights)
                optimizer.t = 0
            elif isinstance(optimizer, SGD):
                optimizer.learning_rate = 0.01
        
        save_params(neural_network, filename)

    # Load data from the file
    data = np.load(filename)
    
    # Update the neural network parameters with the loaded data
    for i, layer in enumerate(neural_network.layers):
        if hasattr(layer, 'weights') and f'weights_layer_{i}' in data:
            layer.weights = data[f'weights_layer_{i}']
        if hasattr(layer, 'biases') and f'biases_layer_{i}' in data:
            layer.biases = data[f'biases_layer_{i}']
    
    if hasattr(neural_network, 'optimizer'):
        from .optimizers import SGD, Adam
        optimizer = neural_network.optimizer
        if isinstance(optimizer, Adam):
            if 'adam_m' in data:
                optimizer.m = data['adam_m']
            else:
                optimizer.m = np.zeros_like(neural_network.layers[0].weights)
            if 'adam_v' in data:
                optimizer.v = data['adam_v']
            else:
                optimizer.v = np.zeros_like(neural_network.layers[0].weights)
            if 'adam_t' in data:
                optimizer.t = data['adam_t']
            else:
                optimizer.t = 0
        elif isinstance(optimizer, SGD):
            if 'sgd_lr' in data:
                optimizer.learning_rate = data['sgd_lr']
            else:
                optimizer.learning_rate = 0.01



def save_results(results, filename='results.json'):
    with open(filename, 'w') as f:
        json.dump(results, f)

def save_trains_results(results, filename='train_results.json'):
    with open(filename, 'w') as f:
        json.dump(results, f)

# Fast Fourier Transform
def fft(func: function):
    pass