
import numpy as np

# Activation functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Loss function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    return (y_pred - y_true) / y_true.shape[0]

# Simple neural network class
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1. / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.Z2
        return self.A2

    def backward(self, X, y_true, y_pred):
        dA2 = mse_derivative(y_true, y_pred)
        dW2 = self.A1.T @ dA2
        db2 = np.sum(dA2, axis=0, keepdims=True)
        dA1 = dA2 @ self.W2.T * relu_derivative(self.Z1)
        dW1 = X.T @ dA1
        db1 = np.sum(dA1, axis=0, keepdims=True)
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train_step(self, X, y):
        y_pred = self.forward(X)
        loss = mean_squared_error(y, y_pred)
        self.backward(X, y, y_pred)
        return loss
