import numpy as np

class MeanSquaredError:
    # Compute the mean squared error loss
    # predictions: Model predictions, targets: True labels
    def forward(self, predictions, targets):
        return np.mean((predictions - targets) ** 2)

    # Compute the gradient of the loss with respect to the predictions
    # predictions: Model predictions, targets: True labels
    def backward(self, predictions, targets):
        return 2 * (predictions - targets) / targets.size
    
class CrossEntropyLoss:
    def forward(self, predictions, labels):
        # Assuming predictions are already logits (raw output values from the network)
        probabilities = self.softmax(predictions)
        # Convert labels to one-hot format
        one_hot_labels = np.zeros_like(probabilities)
        one_hot_labels[np.arange(len(labels)), labels] = 1

        # Calculate the loss
        loss = -np.mean(np.sum(one_hot_labels * np.log(probabilities + 1e-9), axis=1))  # Adding to avoid log(0)
        return loss

    def backward(self, predictions, labels):
        # Gradient
        probabilities = self.softmax(predictions)
        one_hot_labels = np.zeros_like(probabilities)
        one_hot_labels[np.arange(len(labels)), labels] = 1
        dA = (probabilities - one_hot_labels) / len(labels)
        return dA
    
    def softmax(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

