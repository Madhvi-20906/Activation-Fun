import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1.0, 0.0)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(np.clip(x, -500, 500)) - 1))

def elu_derivative(x, alpha=1.0):
    return np.where(x > 0, 1.0, elu(x, alpha) + alpha)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def gelu_derivative(x):
    # Approximation of derivative
    cdf = 0.5 * (1.0 + np.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))))
    pdf = np.exp(-0.5 * (x ** 2)) / np.sqrt(2 * np.pi)
    return cdf + x * pdf

def swish(x, beta=1.0):
    return x * sigmoid(beta * x)

def swish_derivative(x, beta=1.0):
    s = sigmoid(beta * x)
    sw = x * s
    return beta * sw + s * (1 - beta * sw)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

# Softmax derivative is a Jacobian, not usually plotted the same way in 2D
# but for visualization purposes on a single dimension:
def softmax_derivative_1d(x):
    s = softmax(x)
    # diagonal of jacobian
    return s * (1 - s)

ACTIVATION_FUNCTIONS = {
    "Sigmoid": {"func": sigmoid, "deriv": sigmoid_derivative},
    "Tanh": {"func": tanh, "deriv": tanh_derivative},
    "ReLU": {"func": relu, "deriv": relu_derivative},
    "Leaky ReLU": {"func": leaky_relu, "deriv": leaky_relu_derivative},
    "ELU": {"func": elu, "deriv": elu_derivative},
    "GELU": {"func": gelu, "deriv": gelu_derivative},
    "Swish": {"func": swish, "deriv": swish_derivative},
    "Softmax": {"func": softmax, "deriv": softmax_derivative_1d}
}
