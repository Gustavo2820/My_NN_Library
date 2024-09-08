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

def is_power_of_2(n):
    return (n > 0) and (n & (n - 1)) == 0

def next_power_of_2(n):
    return 1 << (n - 1).bit_length()

def zero_pad(P, length):
    return np.pad(P, (0, length - len(P)), mode='constant')

def fft(P):
    # Pad data if necessary
    n = len(P)
    if not is_power_of_2(n):
        new_length = next_power_of_2(n)
        P = zero_pad(P, new_length)
        n = new_length

    # FFT implementation
    if n <= 1:
        return np.array(P, dtype=np.complex128)
    
    omega = np.exp(-2j * np.pi / n * np.arange(n))
    
    P_even = P[::2]
    P_odd = P[1::2]
    
    y_even = fft(P_even)
    y_odd = fft(P_odd)
    
    y = np.zeros(n, dtype=np.complex128)
    
    for j in range(n // 2):
        y[j] = y_even[j] + omega[j] * y_odd[j]
        y[j + n // 2] = y_even[j] - omega[j] * y_odd[j]
    
    return y

def inverse_fft(P):
    # Pad data if necessary
    n = len(P)
    if not is_power_of_2(n):
        new_length = next_power_of_2(n)
        P = zero_pad(P, new_length)
        n = new_length

    # Bit-reversal permutation
    P_reversed = np.empty_like(P, dtype=complex)
    for i in range(n):
        i_reversed = int(f'{i:0{int(np.log2(n))}b}'[::-1], 2)
        P_reversed[i_reversed] = P[i]

    # FFT computation (same as FFT but with reversed signs for twiddle factors)
    for size in range(2, n + 1, 2):
        half_size = size // 2
        twiddle_factors = np.exp(2j * np.pi * np.arange(half_size) / size)
        
        for i in range(0, n, size):
            for j in range(half_size):
                t = twiddle_factors[j] * P_reversed[i + j + half_size]
                u = P_reversed[i + j]
                P_reversed[i + j] = u + t
                P_reversed[i + j + half_size] = u - t

    return P_reversed / n