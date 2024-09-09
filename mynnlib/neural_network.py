import numpy as np
from .loss_functions import CrossEntropyLoss, MeanSquaredError
from .utils import accuracy, save_params, save_results, load_params, save_trains_results

class NeuralNetwork:
    # Initialize the neural network
    def __init__(self, loss_func):
        self.layers = []  # List to store layers of the network
        self.loss = loss_func 

    def add_layer(self, layer):
        self.layers.append(layer)

    def load_params(self, filename='network_params.npz'):
        load_params(self, filename)

    # Forward pass: Compute the output of the network
    # X: Input data
    def forward(self, X):
        self.activations = []  # List to store activations (a's) of each layer
        self.inputs = X  # Store input data for backpropagation
        for layer in self.layers:
            X = layer.forward(X)  # Forward pass through each layer
            self.activations.append(X)  # Store activations
        return X
    
    # Backward pass: Compute gradients for backpropagation
    # Y: True labels
    def backward(self, Y):
        # Compute gradient of the loss function with respect to the final layer's activation and the true labels
        dA = self.loss.backward(self.activations[-1], Y)
        # Backpropagate through each layer in reverse order
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    # Update parameters of the layers using gradient descent
    # learning_rate: Step size for parameter updates
    def update(self):
        for layer in self.layers:
            layer.update()

    def train(self, inputs, labels, epochs=10, batch_size=32, filename='network_params.npz', save_file='train_results.json'):
        load_params(self, filename)
        initial_loss = 0
        epoch_losses = {}
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = len(inputs) // batch_size
            for i in range(0, len(inputs), batch_size):
                input_batch = inputs[i:i + batch_size]
                label_batch = labels[i:i + batch_size]
            
                # Forward pass
                a = self.forward(input_batch)  
                loss = self.loss.forward(a, label_batch)
                epoch_loss += loss
            
                # Backward pass
                dA = self.loss.backward(a, label_batch)  # Ensure backward returns gradients
                self.backward(dA)
                self.update()
            
                if (i % 500 == 0):
                    print(f"Epoch {epoch+1}   Batch {i}   Loss: {loss}")
        
            avg_loss = epoch_loss / num_batches
            epoch_losses[epoch + 1] = avg_loss
            print(f"Epoch {epoch+1}/{epochs}   Average Loss: {avg_loss}")
        
            if epoch == 0:
                initial_loss = avg_loss
    
        final_loss = avg_loss
        loss_reduction = (initial_loss - final_loss) / initial_loss * 100
        print(f"Training complete. Loss decreased by {loss_reduction:.2f}%")
    
        save_params(self, filename)

        results = {
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'loss_reduction': loss_reduction,
            'epoch_losses': epoch_losses
        }
        save_trains_results(results, save_file)


    def model_accuracy(self, inputs, labels):
        predictions = self.forward(inputs)
        return accuracy(labels, predictions)


    def run(self, inputs, save_file='results.json'):
        predictions = self.forward(inputs)
        results = predictions.toList()
        save_results(results, save_file)
        return predictions