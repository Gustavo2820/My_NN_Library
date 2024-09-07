import pytest
import numpy as np
import json
import os
from mynnlib import *

def test_dense_initialization():
    dense_layer = Dense(3, 2, activation=None, optimizer=SGD(learning_rate=0.01))
    assert dense_layer.weights.shape == (3, 2)
    assert dense_layer.biases.shape == (1, 2)

def test_forward_pass():
    dense_layer = Dense(3, 2, activation=None, optimizer=SGD(learning_rate=0.01))
    inputs = np.array([[1, 2, 3], [4, 5, 6]])
    output = dense_layer.forward(inputs)
    assert output.shape == (2, 2) 

def test_backward_pass():
    dense_layer = Dense(3, 2, activation=None, optimizer=SGD(learning_rate=0.01))
    inputs = np.array([[1, 2, 3], [4, 5, 6]])
    output = dense_layer.forward(inputs)
    dA = np.array([[1, 0], [0, 1]])
    dA_prev = dense_layer.backward(dA)
    assert dA_prev.shape == inputs.shape

def test_loss_and_accuracy():
    loss_func = CrossEntropyLoss()
    predictions = np.array([[0.1, 0.9], [0.2, 0.8]])
    labels = np.array([1, 1])  # Labels should be single integers, not arrays
    loss = loss_func.forward(predictions, labels)
    assert np.isclose(loss, 0.4, atol=0.01)  # Ensure the expected value is correct

def test_training():
    nn = NeuralNetwork(loss_func=CrossEntropyLoss())
    nn.add_layer(Dense(3, 2, activation=None, optimizer=SGD(learning_rate=0.01)))
    nn.add_layer(Dense(2, 2, activation=None, optimizer=SGD(learning_rate=0.01)))
    
    inputs = np.array([[1, 2, 3], [4, 5, 6]])
    labels = np.array([[0, 1], [0, 1]])  # Verifique se isso está correto para sua configuração

    nn.train(inputs, labels, epochs=1, batch_size=2)
    
def test_save_and_load_params():
    nn = NeuralNetwork(loss_func=CrossEntropyLoss())
    nn.add_layer(Dense(3, 2, activation=None, optimizer=SGD(learning_rate=0.01)))
    nn.add_layer(Dense(2, 2, activation=None, optimizer=SGD(learning_rate=0.01)))

    original_weights = nn.layers[0].weights.copy()
    original_biases = nn.layers[0].biases.copy()
    
    save_params(nn, 'test_params.npz')
    
    nn.layers[0].weights = np.random.randn(3, 2) * 0.01
    nn.layers[0].biases = np.random.randn(1, 2) * 0.01
    
    load_params(nn, 'test_params.npz')
    
    assert np.allclose(nn.layers[0].weights, original_weights)
    assert np.allclose(nn.layers[0].biases, original_biases)
 
    import os
    os.remove('test_params.npz')

def test_save_results():
    nn = NeuralNetwork(loss_func=CrossEntropyLoss())
    nn.add_layer(Dense(3, 2, activation=None, optimizer=SGD(learning_rate=0.01)))

    inputs = np.array([[1, 2, 3], [4, 5, 6]])
    labels = np.array([[0, 1], [0, 1]])

    nn.train(inputs, labels, epochs=1, batch_size=2, save_file='test_results.json')

    with open('test_results.json', 'r') as f:
        results = json.load(f)

    assert 'epoch_losses' in results
    assert isinstance(results['epoch_losses'], dict)
    assert all(isinstance(key, str) for key in results['epoch_losses'].keys()) 
    assert all(isinstance(value, (int, float)) for value in results['epoch_losses'].values())

    os.remove('test_results.json')
