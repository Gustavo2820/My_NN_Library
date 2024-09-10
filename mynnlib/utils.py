import numpy as np
import os
import json

def accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))

def save_params(neural_network, filename='network_params.npz'):
    from .layers import Dense, BatchNormalization, Conv2D, Dropout, Flatten, MaxPooling
    params = {}
    
    for i, layer in enumerate(neural_network.layers):
        if isinstance(layer, Dense):
            params[f'weights_layer_{i}'] = layer.weights
            params[f'biases_layer_{i}'] = layer.biases
        
        elif isinstance(layer, Conv2D):
            params[f'weights_layer_{i}'] = layer.weights
            params[f'biases_layer_{i}'] = layer.biases
        
        elif isinstance(layer, BatchNormalization):
            params[f'gamma_layer_{i}'] = layer.gamma
            params[f'beta_layer_{i}'] = layer.beta

    # Save optimizer parameters if available
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

def load_params(model, filename='network_params.npz'):
    from .layers import Dense, BatchNormalization, Conv2D, Dropout, Flatten, MaxPooling
    """
    Load the parameters of the neural network from a file.

    Parameters:
    - model: The neural network model containing layers.
    - filename: The name of the file from which to load the parameters.
    """
    if not os.path.exists(filename):
        print(f"File {filename} not found. Initializing parameters with default values.")
        
        for layer in model.layers:
            if isinstance(layer, Dense):
                layer.weights = np.random.randn(*layer.weights.shape) * 0.01
                layer.biases = np.zeros_like(layer.biases)
            
            elif isinstance(layer, Dropout):
                pass
            
            elif isinstance(layer, BatchNormalization):
                layer.gamma = np.ones_like(layer.gamma)
                layer.beta = np.zeros_like(layer.beta)
                layer.running_mean = np.zeros_like(layer.running_mean)
                layer.running_var = np.ones_like(layer.running_var)
            
            elif isinstance(layer, Conv2D):
                layer.weights = np.random.randn(*layer.weights.shape) * 0.01
                layer.biases = np.zeros_like(layer.biases)
            
            elif isinstance(layer, Flatten):
                pass
            
            elif isinstance(layer, MaxPooling):
                pass
        
        return
    
    # Load parameters from file
    with open(filename, 'r') as file:
        params = json.load(file)
    
    layer_index = 0
    for layer_name, layer_params in params.items():
        layer = model.layers[layer_index]
        if isinstance(layer, Dense):
            if 'weights' in layer_params:
                layer.weights = np.array(layer_params['weights'])
            if 'biases' in layer_params:
                layer.biases = np.array(layer_params['biases'])
        
        elif isinstance(layer, Dropout):
            pass
        
        elif isinstance(layer, BatchNormalization):
            if 'gamma' in layer_params:
                layer.gamma = np.array(layer_params['gamma'])
            if 'beta' in layer_params:
                layer.beta = np.array(layer_params['beta'])
            if 'running_mean' in layer_params:
                layer.running_mean = np.array(layer_params['running_mean'])
            if 'running_var' in layer_params:
                layer.running_var = np.array(layer_params['running_var'])
        
        elif isinstance(layer, Conv2D):
            if 'weights' in layer_params:
                layer.weights = np.array(layer_params['weights'])
            if 'biases' in layer_params:
                layer.biases = np.array(layer_params['biases'])
        
        elif isinstance(layer, Flatten):
            pass
        
        elif isinstance(layer, MaxPooling):
            pass
        
        layer_index += 1


def save_results(results, filename='results.json'):
    with open(filename, 'w') as f:
        json.dump(results, f)

def save_trains_results(results, filename='train_results.json'):
    with open(filename, 'w') as f:
        json.dump(results, f)

def is_power_of_2(n):
    return (n > 0) and (n & (n - 1)) == 0

def next_power_of_2(n):
    return 1 << (n - 1).bit_length()

import numpy as np

import numpy as np

def zero_pad(P, pad_width):
    """
    Pad an array with zeros.

    Parameters:
    - P: The input array to be padded.
    - pad_width: A tuple specifying the padding for each dimension, e.g., ((top, bottom), (left, right)).

    Returns:
    - The padded array.
    """
    if not isinstance(pad_width, tuple) or len(pad_width) != P.ndim:
        raise ValueError("pad_width must be a tuple with the same length as the number of dimensions of P.")
    return np.pad(P, pad_width, mode='constant')
