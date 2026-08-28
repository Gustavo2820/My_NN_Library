import json

import numpy as np
import pytest

from mynnlib import (
    Adam,
    BatchNormalization,
    Conv2D,
    CrossEntropyLoss,
    Dense,
    Dropout,
    Flatten,
    MaxPooling,
    MeanSquaredError,
    NeuralNetwork,
    ReLU,
    SGD,
    Sigmoid,
    Tanh,
    accuracy,
    is_power_of_2,
    load_params,
    next_power_of_2,
    save_params,
    zero_pad,
)


def test_dense_forward_and_backward_match_hand_calculation():
    layer = Dense(2, 2, optimizer=SGD(learning_rate=0.1))
    layer.weights = np.array([[1.0, 2.0], [3.0, 4.0]])
    layer.biases = np.array([[0.5, -0.5]])
    inputs = np.array([[1.0, 2.0], [3.0, 4.0]])

    output = layer.forward(inputs)
    upstream_gradient = np.array([[1.0, -1.0], [2.0, 3.0]])
    input_gradient = layer.backward(upstream_gradient)

    np.testing.assert_allclose(output, [[7.5, 9.5], [15.5, 21.5]])
    np.testing.assert_allclose(layer.dW, [[7.0, 8.0], [10.0, 10.0]])
    np.testing.assert_allclose(layer.dB, [[3.0, 2.0]])
    np.testing.assert_allclose(input_gradient, [[-1.0, -1.0], [8.0, 18.0]])


def test_dense_weight_gradient_matches_finite_difference():
    layer = Dense(2, 1, optimizer=SGD())
    layer.weights = np.array([[0.2], [-0.3]])
    layer.biases = np.array([[0.1]])
    inputs = np.array([[1.5, -2.0], [-1.0, 3.0]])
    upstream_gradient = np.array([[0.7], [-0.4]])

    layer.forward(inputs)
    layer.backward(upstream_gradient)

    epsilon = 1e-6
    numerical_gradient = np.zeros_like(layer.weights)
    for row in range(layer.weights.shape[0]):
        original = layer.weights[row, 0]
        layer.weights[row, 0] = original + epsilon
        plus = np.sum(layer.forward(inputs) * upstream_gradient)
        layer.weights[row, 0] = original - epsilon
        minus = np.sum(layer.forward(inputs) * upstream_gradient)
        layer.weights[row, 0] = original
        numerical_gradient[row, 0] = (plus - minus) / (2 * epsilon)

    np.testing.assert_allclose(layer.dW, numerical_gradient, rtol=1e-6, atol=1e-6)


def test_sgd_updates_weights_and_biases():
    layer = Dense(2, 1, optimizer=SGD(learning_rate=0.25))
    layer.weights = np.array([[1.0], [2.0]])
    layer.biases = np.array([[3.0]])
    layer.dW = np.array([[4.0], [-2.0]])
    layer.dB = np.array([[8.0]])

    layer.update()

    np.testing.assert_allclose(layer.weights, [[0.0], [2.5]])
    np.testing.assert_allclose(layer.biases, [[1.0]])


@pytest.mark.parametrize(
    ("activation", "values", "expected"),
    [
        (ReLU(), np.array([[-2.0, 0.0, 3.0]]), np.array([[0.0, 0.0, 3.0]])),
        (Sigmoid(), np.array([[0.0]]), np.array([[0.5]])),
        (Tanh(), np.array([[0.0]]), np.array([[0.0]])),
    ],
)
def test_activations_forward(activation, values, expected):
    np.testing.assert_allclose(activation.forward(values), expected)


def test_activation_backward_values():
    np.testing.assert_allclose(
        ReLU().backward(np.ones((1, 3)), np.array([[-1.0, 0.0, 1.0]])),
        [[0.0, 0.0, 1.0]],
    )
    np.testing.assert_allclose(
        Sigmoid().backward(np.array([[2.0]]), np.array([[0.0]])),
        [[0.5]],
    )
    np.testing.assert_allclose(
        Tanh().backward(np.array([[2.0]]), np.array([[0.0]])),
        [[2.0]],
    )


def test_mean_squared_error_value_and_gradient():
    predictions = np.array([[1.0], [3.0]])
    targets = np.array([[2.0], [1.0]])
    loss = MeanSquaredError()

    assert loss.forward(predictions, targets) == pytest.approx(2.5)
    np.testing.assert_allclose(loss.backward(predictions, targets), [[-1.0], [2.0]])


def test_cross_entropy_matches_softmax_reference_and_gradient_contract():
    logits = np.array([[2.0, 1.0], [0.0, 2.0]])
    labels = np.array([0, 1])
    loss = CrossEntropyLoss()

    probabilities = loss.softmax(logits)
    expected_loss = -np.mean(np.log(probabilities[np.arange(2), labels]))
    expected_gradient = (probabilities - np.eye(2)[labels]) / 2

    assert loss.forward(logits, labels) == pytest.approx(expected_loss)
    np.testing.assert_allclose(loss.backward(probabilities, labels), expected_gradient)
    np.testing.assert_allclose(np.sum(probabilities, axis=1), [1.0, 1.0])


def test_network_training_reduces_mse_and_writes_artifacts(tmp_path, capsys):
    np.random.seed(7)
    inputs = np.array([[-1.0], [0.0], [1.0]])
    targets = 2.0 * inputs + 1.0
    model = NeuralNetwork(MeanSquaredError())
    model.add_layer(Dense(1, 1, optimizer=SGD(learning_rate=0.05)))
    initial_loss = model.loss.forward(model.forward(inputs), targets)
    params_path = tmp_path / "model.npz"
    results_path = tmp_path / "training.json"

    model.train(
        inputs,
        targets,
        epochs=80,
        batch_size=3,
        filename=params_path,
        save_file=results_path,
    )

    final_loss = model.loss.forward(model.forward(inputs), targets)
    results = json.loads(results_path.read_text(encoding="utf-8"))

    assert final_loss < initial_loss * 0.01
    assert params_path.exists()
    assert results["initial_loss"] == pytest.approx(initial_loss)
    # train registra a perda antes da última atualização; a avaliação acima
    # acontece depois dessa atualização e, por isso, tende a ser menor.
    assert results["final_loss"] == pytest.approx(results["epoch_losses"]["80"])
    assert results["final_loss"] > final_loss
    assert len(results["epoch_losses"]) == 80
    assert "Training complete" in capsys.readouterr().out


def test_save_and_load_preserves_dense_parameters(tmp_path):
    model = NeuralNetwork(MeanSquaredError())
    first = Dense(2, 3, optimizer=SGD())
    second = Dense(3, 1, optimizer=SGD())
    model.add_layer(first)
    model.add_layer(second)
    first.weights = np.arange(6, dtype=float).reshape(2, 3)
    first.biases = np.array([[1.0, 2.0, 3.0]])
    second.weights = np.array([[4.0], [5.0], [6.0]])
    second.biases = np.array([[7.0]])
    path = tmp_path / "parameters.npz"

    save_params(model, path)
    first.weights.fill(-1)
    first.biases.fill(-1)
    second.weights.fill(-1)
    second.biases.fill(-1)
    load_params(model, path)

    np.testing.assert_allclose(first.weights, [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    np.testing.assert_allclose(first.biases, [[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(second.weights, [[4.0], [5.0], [6.0]])
    np.testing.assert_allclose(second.biases, [[7.0]])


def test_load_params_requires_an_existing_file(tmp_path):
    model = NeuralNetwork(MeanSquaredError())

    with pytest.raises(FileNotFoundError):
        load_params(model, tmp_path / "missing.npz")


def test_shape_layers_and_dropout():
    image = np.array([[[[1.0], [3.0]], [[2.0], [4.0]]]])
    flattened = Flatten()
    pooling = MaxPooling(pool_size=(2, 2), stride=2)
    dropout = Dropout(rate=0.5)

    flat = flattened.forward(image)
    np.testing.assert_allclose(flat, [[1.0, 3.0, 2.0, 4.0]])
    np.testing.assert_allclose(flattened.backward(np.ones((1, 4))), np.ones_like(image))
    np.testing.assert_allclose(pooling.forward(image), [[[[4.0]]]])
    np.testing.assert_allclose(pooling.backward(np.array([[[[2.0]]]])), [[[[0.0], [0.0]], [[0.0], [2.0]]]])

    np.random.seed(4)
    dropped = dropout.forward(np.ones((4, 4)))
    assert set(np.unique(dropped)).issubset({0.0, 2.0})
    np.testing.assert_allclose(dropout.forward(image, training=False), image)


def test_batch_normalization_training_and_inference():
    layer = BatchNormalization(momentum=0.0)
    inputs = np.array([[1.0, 3.0], [3.0, 7.0]])

    output = layer.forward(inputs)
    inference = layer.forward(inputs, training=False)

    np.testing.assert_allclose(np.mean(output, axis=0), [0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(inference, output)


def test_utils_and_accuracy():
    assert is_power_of_2(8)
    assert not is_power_of_2(0)
    assert not is_power_of_2(10)
    assert next_power_of_2(1) == 1
    assert next_power_of_2(9) == 16
    with pytest.raises(ValueError):
        next_power_of_2(0)
    np.testing.assert_allclose(zero_pad(np.array([[1]]), ((1, 1), (2, 2))), [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]])
    with pytest.raises(ValueError):
        zero_pad(np.array([[1]]), ((1, 1),))
    assert accuracy(np.eye(2)[[0, 1]], np.array([[0.9, 0.1], [0.2, 0.8]])) == 1.0


@pytest.mark.xfail(strict=True, raises=AttributeError, reason="run usa toList(), que não existe em ndarray")
def test_run_serializes_numpy_predictions(tmp_path):
    model = NeuralNetwork(MeanSquaredError())
    model.add_layer(Dense(1, 1, optimizer=SGD()))
    model.run(np.array([[1.0]]), save_file=tmp_path / "results.json")


@pytest.mark.xfail(strict=True, raises=TypeError, reason="Conv2D não inicializa weights antes do forward")
def test_conv2d_initializes_parameters_on_first_forward():
    Conv2D(filters=1, kernel_size=(2, 2)).forward(np.ones((1, 3, 3, 1)))


@pytest.mark.xfail(strict=True, raises=AttributeError, reason="Adam espera uma rede, mas Dense o chama com a própria camada")
def test_adam_is_usable_as_dense_optimizer():
    layer = Dense(1, 1, optimizer=Adam())
    layer.forward(np.array([[1.0]]))
    layer.backward(np.array([[1.0]]))
    layer.update()
