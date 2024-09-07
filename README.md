# MyNNLib

MyNNLib is a lightweight Python library designed for building and training neural networks. This library includes basic neural network components such as layers, activation functions, and loss functions. It is a great tool for learning about neural networks and experimenting with custom architectures.

## Features

- **Layers**: Dense layers with customizable activation functions and optimizers.
- **Activation Functions**: ReLU, Sigmoid, and more.
- **Loss Functions**: Mean Squared Error and Cross-Entropy Loss with built-in support for softmax.
- **Optimizers**: Stochastic Gradient Descent (SGD) and Adaptive Moment Estimation (Adam)
- **Utilities**: Functions for saving and loading model weights.

## Installation

To install MyNNLib, clone the repository and use pip to install it:

```
git clone https://github.com/Gustavo2820/My_NN_Library.git
cd My_NN_Library
pip install .
```

## Usage

### Basic Example

Here's a basic example of how to use MyNNLib to build and train a neural network:

```
import numpy as np
from mynnlib.neural_network import NeuralNetwork
from mynnlib.layers import Dense
from mynnlib.activations import ReLU, Sigmoid
from mynnlib.loss_functions import CrossEntropyLoss
from mynnlib.optimizers import SGD

# Create a neural network
nn = NeuralNetwork(loss_func=CrossEntropyLoss())

# Add layers
nn.add_layer(Dense(3, 2, activation=ReLU(), optimizer=SGD(learning_rate=0.01)))
nn.add_layer(Dense(2, 2, activation=Sigmoid(), optimizer=SGD(learning_rate=0.01)))

# Generate random data
inputs = np.array([[1, 2, 3], [4, 5, 6]])
labels = np.array([1, 1])

# Train the network
nn.train(inputs, labels, epochs=10, batch_size=2, save_file='train_results.json')
```

