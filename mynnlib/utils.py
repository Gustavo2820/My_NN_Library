import json
import os

import numpy as np


def accuracy(y_true, y_pred):
    """
    Calculate classification accuracy.

    This function assumes that both `y_true` and `y_pred` are represented
    as arrays where each row contains scores or one-hot encoded values for
    each class.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels, usually in one-hot encoded format.

    y_pred : np.ndarray
        Model predictions or class scores.

    Returns
    -------
    float
        Fraction of correctly classified samples, between 0 and 1.
    """
    true_classes = np.argmax(y_true, axis=1)
    predicted_classes = np.argmax(y_pred, axis=1)

    return np.mean(true_classes == predicted_classes)


def save_params(neural_network, filename="network_params.npz"):
    """
    Save all trainable parameters of a neural network to a NumPy `.npz` file.

    The parameters are stored using keys based on the layer index. For example:

        weights_layer_0
        biases_layer_0
        gamma_layer_2
        beta_layer_2

    Currently supported layers:
    - Dense
    - Conv2D
    - BatchNormalization

    Parameters
    ----------
    neural_network : NeuralNetwork
        Neural network instance containing the layers to be saved.

    filename : str, optional
        Output file path. Defaults to ``network_params.npz``.

    Notes
    -----
    This function only stores trainable layer parameters. Model architecture
    itself is not serialized, so the network structure must already exist
    before loading these parameters again.
    """
    from .layers import BatchNormalization, Conv2D, Dense

    params = {}

    for i, layer in enumerate(neural_network.layers):
        if isinstance(layer, Dense):
            params[f"weights_layer_{i}"] = layer.weights
            params[f"biases_layer_{i}"] = layer.biases

        elif isinstance(layer, Conv2D):
            params[f"weights_layer_{i}"] = layer.weights
            params[f"biases_layer_{i}"] = layer.biases

        elif isinstance(layer, BatchNormalization):
            params[f"gamma_layer_{i}"] = layer.gamma
            params[f"beta_layer_{i}"] = layer.beta

    np.savez(filename, **params)


def load_params(model, filename):
    """
    Load previously saved neural network parameters from a `.npz` file.

    The model architecture must already exist and be compatible with the
    parameters stored in the file.

    Parameters
    ----------
    model : NeuralNetwork
        Neural network instance whose layers will receive the loaded
        parameters.

    filename : str
        Path to the `.npz` parameter file.

    Raises
    ------
    FileNotFoundError
        If the requested parameter file does not exist.

    Notes
    -----
    This function expects files created by :func:`save_params`.

    Loading parameters does not reconstruct the model architecture.
    The caller must create the network with the correct layer structure
    before calling this function.
    """
    from .layers import BatchNormalization, Conv2D, Dense

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Parameter file not found: {filename}"
        )

    with np.load(filename, allow_pickle=False) as params:
        for i, layer in enumerate(model.layers):
            if isinstance(layer, Dense):
                weights_key = f"weights_layer_{i}"
                biases_key = f"biases_layer_{i}"

                if weights_key in params:
                    layer.weights = params[weights_key].copy()

                if biases_key in params:
                    layer.biases = params[biases_key].copy()

            elif isinstance(layer, Conv2D):
                weights_key = f"weights_layer_{i}"
                biases_key = f"biases_layer_{i}"

                if weights_key in params:
                    layer.weights = params[weights_key].copy()

                if biases_key in params:
                    layer.biases = params[biases_key].copy()

            elif isinstance(layer, BatchNormalization):
                gamma_key = f"gamma_layer_{i}"
                beta_key = f"beta_layer_{i}"

                if gamma_key in params:
                    layer.gamma = params[gamma_key].copy()

                if beta_key in params:
                    layer.beta = params[beta_key].copy()


def save_results(results, filename="results.json"):
    """
    Save inference results to a JSON file.

    Parameters
    ----------
    results : object
        JSON-serializable object containing prediction results.

    filename : str, optional
        Output file path. Defaults to ``results.json``.
    """
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)


def save_trains_results(results, filename="train_results.json"):
    """
    Save training results or metrics to a JSON file.

    Parameters
    ----------
    results : object
        JSON-serializable training information, such as losses,
        accuracies or epoch statistics.

    filename : str, optional
        Output file path. Defaults to ``train_results.json``.
    """
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)


def is_power_of_2(n):
    """
    Check whether an integer is a power of two.

    Parameters
    ----------
    n : int
        Integer to test.

    Returns
    -------
    bool
        True if `n` is a positive power of two, otherwise False.

    Examples
    --------
    >>> is_power_of_2(8)
    True

    >>> is_power_of_2(10)
    False
    """
    return n > 0 and (n & (n - 1)) == 0


def next_power_of_2(n):
    """
    Return the smallest power of two greater than or equal to `n`.

    Parameters
    ----------
    n : int
        Positive integer.

    Returns
    -------
    int
        Smallest power of two greater than or equal to `n`.

    Examples
    --------
    >>> next_power_of_2(5)
    8

    >>> next_power_of_2(16)
    16
    """
    if n <= 0:
        raise ValueError("n must be a positive integer.")

    return 1 << (n - 1).bit_length()


def zero_pad(array, pad_width):
    """
    Pad an array with zeros.

    This is a thin wrapper around ``numpy.pad`` using constant zero padding.

    Parameters
    ----------
    array : np.ndarray
        Input array.

    pad_width : tuple
        Padding configuration. Its length must match the number of dimensions
        in the input array.

        Example for a 2D array:

            ((top, bottom), (left, right))

    Returns
    -------
    np.ndarray
        Zero-padded array.

    Raises
    ------
    ValueError
        If `pad_width` is not a tuple or does not contain one entry for each
        dimension of the input array.
    """
    if not isinstance(pad_width, tuple):
        raise ValueError("pad_width must be a tuple.")

    if len(pad_width) != array.ndim:
        raise ValueError(
            "pad_width must contain one padding specification "
            "for each dimension of the input array."
        )

    return np.pad(array, pad_width, mode="constant")
