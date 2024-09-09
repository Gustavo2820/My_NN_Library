import numpy as np
from .optimizers import SGD, Adam
from .utils import save_params, load_params, zero_pad, next_power_of_2, fft, inverse_fft

class Dense:
    def __init__(self, input_size, output_size, activation=None, optimizer=SGD(learning_rate=0.01)):
        """
        Initializes the dense layer.

        Parameters:
        - input_size: Size of the input vector.
        - output_size: Size of the output vector..
        - activation: Activation function to apply after convolution.
        - optimizer: Optimizer for the layer parameters.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) * 0.01 # Initialize weights
        self.biases = np.zeros((1, output_size))  # Initialize biases
        self.activation = activation  # Activation function (if any)
        self.optimizer = optimizer

    def forward(self, inputs):
        """
        Performs the forward pass of the dense layer.

        Parameters:
        - inputs: Vector of inputs.

        Returns:
        - a: activated weighted sum.
        """
        self.inputs = inputs  # Store inputs
        self.z = np.dot(inputs, self.weights) + self.biases  # Compute weighted sum plus biases
        if self.activation:  # Apply activation function if provided
            self.a = self.activation.forward(self.z)
        else:
            self.a = self.z  # No activation
        return self.a
    
    def backward(self, dA):
        """
        Performs the backward pass of the dense layer.

        Parameters:
        - dA: gradient w.r.t activated weighted sum of the previous layer. (In backpropagation process)

        Returns:
        - dA_prev: gradient of the cost function w.r.t the input of the convolutional layer.
        """
        if self.activation:
            dZ = self.activation.backward(dA, self.z)
        else:
            dZ = dA
        self.dW = np.dot(self.inputs.T, dZ)  # (input_size, batch_size) dot (batch_size, output_size) -> (input_size, output_size)

        self.dB = np.sum(dZ, axis=0, keepdims=True)  # (batch_size, output_size) -> (1, output_size)
    
        dA_prev = np.dot(dZ, self.weights.T)  # (batch_size, output_size) dot (output_size, input_size) -> (batch_size, input_size)

        return dA_prev

    def update(self):
        self.optimizer.update(self)

class Dropout:
    def __init__(self, rate):
        self.rate = rate

    def forward(self, inputs, training=True):
        if not training:
            return inputs
        self.mask = np.random.binomial(1, 1 - self.rate, size=inputs.shape) / (1 - self.rate)
        return inputs * self.mask
    
    def backward(self, dA):
        return dA * self.mask
    
class BatchNormalization:
    def __init__(self, momentum=0.9, epsilon=1e-8):
        self.momentum = momentum
        self.epsilon = epsilon
        self.gamma = np.ones(1)  # Initialize scale parameter
        self.beta = np.zeros(1)  # Initialize shift parameter
        self.running_mean = 0
        self.running_var = 0

    def forward(self, inputs, training=True):
        if training:
            # Calculate mean and variance
            self.mean = np.mean(inputs, axis=0)
            self.variance = np.var(inputs, axis=0)

            # Update running mean and variance
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.variance

            # Normalize inputs
            self.normalized_inputs = (inputs - self.mean) / np.sqrt(self.variance + self.epsilon)
            self.inputs = inputs

            # Scale and shift
            out = self.gamma * self.normalized_inputs + self.beta
            return out
        else:
            # During inference, use running mean and variance
            normalized_inputs = (inputs - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            return self.gamma * normalized_inputs + self.beta

    def backward(self, dA):
        N, D = dA.shape
        
        # Gradient with respect to gamma (scale parameter) and beta (shift parameter)
        dGamma = np.sum(dA * self.normalized_inputs, axis=0)
        dBeta = np.sum(dA, axis=0)

        # Gradient w.r.t. the normalized inputs
        dNormalized = dA * self.gamma
        
        # Gradient w.r.t. the variance
        dVar = np.sum(dNormalized * (self.inputs - self.mean) * -0.5 * np.power(self.variance + self.epsilon, -1.5), axis=0)
        # Gradient w.r.t. the mean
        dMean = np.sum(dNormalized * -1 / np.sqrt(self.variance + self.epsilon), axis=0) + dVar * np.mean(-2 * (self.inputs - self.mean), axis=0)
        # Final gradient w.r.t. inputs
        dX = dNormalized / np.sqrt(self.variance + self.epsilon) + dVar * 2 * (self.inputs - self.mean) / N + dMean / N
        
        return dX

    def update(self):
        # Update gamma and beta if they are being optimized
        if hasattr(self, 'optimizer'):
            self.gamma = self.optimizer.update(self.gamma, self.dGamma)
            self.beta = self.optimizer.update(self.beta, self.dBeta)

class Conv2D:
    def __init__(self, filters, kernel_size, stride=1, padding=0, activation=None, optimizer=SGD(learning_rate=0.01)):
        """
        Initializes the Conv2D layer.

        Parameters:
        - filters: Number of filters (kernels) to apply.
        - kernel_size: Size of the convolutional kernels (tuple, e.g., (3, 3)).
        - stride: Stride (step) of the convolution.
        - padding: Padding to add around the input.
        - activation: Activation function to apply after convolution.
        - optimizer: Optimizer for the layer parameters.
        """
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.optimizer = optimizer
        self.stride = stride
        self.padding = padding
        self.weights = None
        self.biases = None
        self.cache = None

    def init_wb(self, input_shape):
        """
        Initializes the weights and biases of the convolutional layer.

        Parameters:
        - input_shape: Shape of the input data (n_samples, height, width, channels).
        """
        n, input_h, input_w, input_c = input_shape
        kernel_h, kernel_w = self.kernel_size

        # Initialize weights with small random values and biases with zeros
        self.weights = np.random.randn(kernel_h, kernel_w, input_c, self.filters) * 0.01
        self.biases = np.zeros((1, 1, 1, self.filters))

    def forward(self, inputs):
        """
        Performs the forward pass of the convolutional layer using FFT.

        Parameters:
        - inputs: Input data (n_samples, height, width, channels).

        Returns:
        - Output data after applying convolution and activation.
        """
    
        # Apply padding
        if self.padding > 0:
            inputs = np.pad(inputs, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode="constant")

        self.cache = inputs  # Store inputs for backpropagation
        n, input_h, input_w, input_c = inputs.shape
        kernel_h, kernel_w = self.kernel_size
        output_h = (input_h - kernel_h) // self.stride + 1
        output_w = (input_w - kernel_w) // self.stride + 1

        # Calculate the size for FFT
        padded_h = next_power_of_2(input_h + kernel_h - 1)
        padded_w = next_power_of_2(input_w + kernel_w - 1)

        # Pad the input and the kernel
        padded_inputs = zero_pad(inputs, (padded_h, padded_w, input_c))
        padded_weights = np.zeros((padded_h, padded_w, self.filters))
    
        # Place each filter into a 2D array of the same size as padded_inputs
        for i in range(self.filters):
            padded_weights[:self.kernel_size[0], :self.kernel_size[1], i] = self.weights[:, :, :, i]

        # Apply FFT
        fft_inputs = fft(padded_inputs)
        fft_weights = fft(padded_weights)
    
        # Multiply in the frequency domain
        fft_output = fft_inputs * fft_weights
    
        # Apply IFFT
        convolved = inverse_fft(fft_output)
    
        # Crop to the original output size
        convolved = convolved[:output_h, :output_w, :]
    
        if self.activation:
            convolved = self.activation.forward(convolved)
    
        return convolved

    
    def backward(self, dA):
        """
        Performs the backward pass for the convolutional layer using FFT.

        Parameters:
        - dA: Gradient of the cost function w.r.t activated output from the previous layers.

        Returns:
        - dA_prev: Gradient of the cost function w.r.t the input of the convolutional layer.
        """
        inputs = self.cache
        n, input_h, input_w, input_c = inputs.shape
        kernel_h, kernel_w = self.kernel_size
        output_h, output_w, _ = dA.shape

        # Compute the size for FFT
        padded_h = next_power_of_2(input_h + kernel_h - 1)
        padded_w = next_power_of_2(input_w + kernel_w - 1)

        # Pad the input and dA
        padded_inputs = zero_pad(inputs, (padded_h, padded_w, input_c))
        padded_dA = zero_pad(dA, (padded_h, padded_w, self.filters))

        # Apply FFT
        fft_inputs = fft(padded_inputs)
        fft_dA = fft(padded_dA)

        # Compute gradients
        fft_dW = fft_inputs * fft_dA
        fft_dB = fft_dA

        # Apply IFFT
        dA_prev = inverse_fft(fft_dA)
        dW = inverse_fft(fft_dW)
        dB = np.sum(dA, axis=(0, 1, 2))  # Bias gradients

        # Crop the result to the original input size
        dA_prev = dA_prev[:input_h, :input_w, :]

        return dA_prev


    def update(self):
        self.optimizer.update(self)