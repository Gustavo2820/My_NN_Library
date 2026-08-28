# MyNNLib

[![Version](https://img.shields.io/badge/version-0.1.2-blue.svg)](https://github.com/Gustavo2820/My_NN_Library/releases)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

MyNNLib is a small neural-network library built with NumPy. It is intended for learning and experimentation: layers, losses, activations, and parameter updates are implemented in plain Python so the training loop is easy to inspect.

## What it does

The library lets you assemble a sequential network, run a forward pass, compute a loss, backpropagate gradients, and update parameters.

The most practical path today is a dense network trained with Mean Squared Error (MSE) and SGD. The package also exposes dropout, batch normalization, pooling, flattening, convolution, common activation functions, model-parameter persistence, and a few NumPy helpers.

## Why use it

- Small codebase that is straightforward to read and modify.
- No runtime dependency beyond NumPy.
- Useful for following the mechanics of backpropagation without a large framework.
- Includes tests for dense-layer calculations, numerical gradients, training, persistence, shape layers, activations, and utilities.
- Installable as a normal Python package.

## Install

MyNNLib requires Python 3 and NumPy.

~~~bash
git clone https://github.com/Gustavo2820/My_NN_Library.git
cd My_NN_Library
python -m pip install .
~~~

For local development, install the test dependency too:

~~~bash
python -m pip install -e ".[dev]"
python -m pytest -q
~~~

## Quick start

This example trains a one-layer model for a simple regression problem.

~~~python
import numpy as np

from mynnlib import Dense, MeanSquaredError, NeuralNetwork, SGD

X = np.array([[0.0], [1.0], [2.0], [3.0]])
y = np.array([[1.0], [3.0], [5.0], [7.0]])

model = NeuralNetwork(loss_func=MeanSquaredError())
model.add_layer(Dense(1, 1, optimizer=SGD(learning_rate=0.01)))

model.train(
    X,
    y,
    epochs=200,
    batch_size=4,
    filename="regression_params.npz",
    save_file="regression_results.json",
)

prediction = model.forward(np.array([[4.0]]))
print(prediction)
~~~

The input's first dimension is the batch dimension. For Dense(input_size, output_size), inputs have shape (batch, input_size) and outputs have shape (batch, output_size). With MSE, targets must have the same shape as the model output.

Training writes weights and biases to an NPZ file and loss information to JSON. To load saved parameters, recreate the same architecture before calling load_params.

~~~python
model.load_params("regression_params.npz")
~~~

## Project layout

~~~text
mynnlib/
  __init__.py              Public imports
  neural_network.py        Training loop and sequential network
  layers.py                Dense, convolution, pooling, and shape layers
  activation_functions.py  ReLU, Sigmoid, and Tanh
  loss_functions.py        MSE and cross-entropy losses
  optimizers.py            SGD and Adam
  utils.py                 Persistence, metrics, and helpers
tests/
  test_lib.py              Library test suite
~~~

## Current scope

MyNNLib is a learning project rather than a production ML framework. Dense + MSE + SGD is the supported example to start with.

Some public components are still experimental:

- CrossEntropyLoss is not yet wired correctly into the NeuralNetwork training flow.
- Adam does not yet share the same update interface used by Dense.
- Conv2D parameter initialization and backward updates are incomplete.
- NeuralNetwork.run currently cannot serialize a NumPy result.

These cases are marked in the test suite so their status remains visible.

## Help

- Read the source alongside the tests in [tests/test_lib.py](tests/test_lib.py); they are the best executable examples of the current behavior.
- Search or open an issue on the [GitHub repository](https://github.com/Gustavo2820/My_NN_Library/issues) for questions, bug reports, or ideas.
- For changes, start with [CONTRIBUTING.md](CONTRIBUTING.md).

## Maintainer and contributions

MyNNLib is maintained by [Gustavo Oliveira Longuinho](https://github.com/Gustavo2820).

Contributions are welcome, especially fixes and tests for the experimental components. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## License

This project is released under the [MIT License](LICENSE).

