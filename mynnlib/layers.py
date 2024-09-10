import numpy as np
from .optimizers import SGD, Adam
from .utils import save_params, load_params, zero_pad, next_power_of_2

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
        n_samples, input_h, input_w, input_c = input_shape
        kernel_h, kernel_w = self.kernel_size
        self.weights = np.random.randn(kernel_h, kernel_w, input_c, self.filters) * 0.01
        self.biases = np.zeros((1, 1, 1, self.filters))

    def forward(self, inputs):
        if self.padding > 0:
            inputs = np.pad(inputs, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode="constant")

        self.cache = inputs
        n_samples, input_h, input_w, input_c = inputs.shape
        kernel_h, kernel_w = self.kernel_size
        output_h = (input_h - kernel_h + 2 * self.padding) // self.stride + 1
        output_w = (input_w - kernel_w + 2 * self.padding) // self.stride + 1

        output = np.zeros((n_samples, output_h, output_w, self.filters))

        for i in range(self.filters):
            for j in range(output_h):
                for k in range(output_w):
                    h_start = j * self.stride
                    h_end = h_start + kernel_h
                    w_start = k * self.stride
                    w_end = w_start + kernel_w

                    patch = inputs[:, h_start:h_end, w_start:w_end, :]
                    output[:, j, k, i] = np.sum(patch * self.weights[:, :, :, i], axis=(1, 2, 3)) + self.biases[0, 0, 0, i]

        if self.activation:
            output = self.activation.forward(output)

        return output

    def backward(self, dA):
        inputs = self.cache
        n_samples, input_h, input_w, input_c = inputs.shape
        kernel_h, kernel_w = self.kernel_size
        _, output_h, output_w, _ = dA.shape

        padded_inputs = zero_pad(inputs, ((0, 0), (0, input_h - input_h), (0, input_w - input_w), (0, 0)))
        padded_dA = zero_pad(dA, ((0, 0), (0, input_h - output_h), (0, input_w - output_w), (0, 0)))
        print(f"Padded Inputs shape for backward: {padded_inputs.shape}")
        print(f"Padded dA shape for backward: {padded_dA.shape}")

        fft_inputs = np.fft.fftn(padded_inputs)
        fft_dA = np.fft.fftn(padded_dA)
        print(f"FFT Inputs shape for backward: {fft_inputs.shape}")
        print(f"FFT dA shape for backward: {fft_dA.shape}")

        fft_dW = fft_inputs * fft_dA
        fft_dB = fft_dA

        dA_prev = np.fft.ifftn(fft_dA)
        dW = np.fft.ifftn(fft_dW)
        dB = np.sum(dA, axis=(0, 1, 2))
        print(f"Gradient shape dA_prev: {dA_prev.shape}")
        print(f"Gradient shape dW: {dW.shape}")

        dA_prev = np.real(dA_prev)
        dA_prev = dA_prev[:, :input_h, :input_w, :]
        print(f"dA_prev shape after cropping: {dA_prev.shape}")

        return dA_prev

    def update(self):
        self.optimizer.update(self)
class Flatten:
    def __init__(self):
        self.input_shape = None

    def forward(self, inputs):
        """
        Flatten the input data into a single vector.

        Parameters:
        - inputs: Input data (n_samples, height, width, channels).

        Returns:
        - Flattened data (n_samples, height * width * channels).
        """
        self.input_shape = inputs.shape
        n_samples, height, width, channels = self.input_shape
        return inputs.reshape(n_samples, -1)

    def backward(self, dA):
        """
        Reshape the gradient of the loss w.r.t the output to match the input shape.

        Parameters:
        - dA: Gradient of the loss w.r.t the flattened output.

        Returns:
        - Gradient of the loss w.r.t the input of the Flatten layer.
        """
        n_samples, height, width, channels = self.input_shape
        return dA.reshape(n_samples, height, width, channels)

class MaxPooling:
    def __init__(self, pool_size=(2, 2), stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.cache = None

    def forward(self, inputs):
        (n_samples, input_h, input_w, input_c) = inputs.shape
        (pool_h, pool_w) = self.pool_size
        stride_h = stride_w = self.stride

        output_h = (input_h - pool_h) // stride_h + 1
        output_w = (input_w - pool_w) // stride_w + 1

        pooled = np.zeros((n_samples, output_h, output_w, input_c))
        self.cache = inputs

        for i in range(0, input_h - pool_h + 1, stride_h):
            for j in range(0, input_w - pool_w + 1, stride_w):
                pooled[:, i // stride_h, j // stride_w, :] = np.max(
                    inputs[:, i:i + pool_h, j:j + pool_w, :], axis=(1, 2)
                )
        
        return pooled

    def backward(self, dA):
        (n_samples, input_h, input_w, input_c) = self.cache.shape
        (pool_h, pool_w) = self.pool_size
        stride_h = stride_w = self.stride
        (n_samples, output_h, output_w, _) = dA.shape

        dX = np.zeros_like(self.cache)

        for i in range(0, input_h - pool_h + 1, stride_h):
            for j in range(0, input_w - pool_w + 1, stride_w):
                max_mask = (self.cache[:, i:i + pool_h, j:j + pool_w, :] == np.max(
                    self.cache[:, i:i + pool_h, j:j + pool_w, :], axis=(1, 2), keepdims=True
                ))
                dX[:, i:i + pool_h, j:j + pool_w, :] += max_mask * dA[:, i // stride_h, j // stride_w, :][:, np.newaxis, np.newaxis, :]

        return dX

