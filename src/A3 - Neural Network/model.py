"""Wrapper for Neural Network for classification

Author: Utkarsh Patel (18EC35034)
This module is a part of MIES Coding Assignment #3
"""

import numpy as np
from tqdm import tqdm
from utils import sigmoid

class InternalError(Exception):
    """Exception handling for the model"""
    pass

class NeuralNetwork:
    """Wrapper for Neural Network"""
    def __init__(self,
                 hidden_layers: list = [8],
                 learning_rate: float = 0.01,
                 batch_size: int = 32,
                 max_iter: int = 500,
                 validation_fraction: float = 0.2,
                 random_state: int = 42):
        """Routine to initialize the Neural Network
        -------------------------------------------
        Input:
        :param hidden_layers: list with i-th element representing the number of neurons in the i-th hidden layer
        :param learning_rate: learning rate schedule for weight updates
        :param batch_size: size of mini-batches for stochastic optimizers
        :param max_iter: maximum number of iterations
        :param validation_fraction: proportion of training data to set aside as validation set
        """
        self.layers_dim = [None] + hidden_layers + [None]
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.validation_fraction = validation_fraction
        self.random_state = random_state

        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.error_curve = [0] * max_iter        # Error at each iteration
        self.train_acc = [0] * max_iter
        self.val_acc = [0] * max_iter
        self.n_layers = len(self.layers_dim)     # Number of layers (including input and output)

        # Model parameters
        self.W = [None] * self.n_layers
        self.b = [None] * self.n_layers
        self.Z = [None] * self.n_layers
        self.A = [None] * self.n_layers

        self.dW = [None] * self.n_layers
        self.db = [None] * self.n_layers
        self.dZ = [None] * self.n_layers
        self.dA = [None] * self.n_layers

    def fit(self,
            x: np.ndarray,
            y: np.ndarray):
        """Routine to train the Neural Network
        --------------------------------------
        Input:
        :param x: training instances
        :param y: training labels (one-hot encoded)
        """
        n_samples = x.shape[0]
        n_partition = int(n_samples * self.validation_fraction)
        while (n_samples - n_partition) % self.batch_size:
            n_partition += 1
        self.layers_dim[0] = x.shape[1]
        self.layers_dim[-1] = y.shape[1]
        self._init_weights()
        self.x_val = x[:n_partition]
        self.y_val = y[:n_partition]
        self.x_train = x[n_partition:]
        self.y_train = y[n_partition:]
        self._run()

    def score(self,
              x: np.ndarray,
              y: np.ndarray) -> float:
        """Routine to test the Neural Network
        -------------------------------------
        :param x: Test instances
        :param y: Test labels (one-hot encoded)
        """
        if x.shape[1] != self.layers_dim[0]:
            raise InternalError(f'Expected n_features to be {self.layers_dim[0]}, got {x.shape[1]}.')
        y_pred = self.predict(x)
        return self._calc_accuracy(y, y_pred)

    def predict(self,
                x: np.ndarray) -> np.ndarray:
        """Routine to get predicted labels
        ----------------------------------
        Input:
        :param x: test instances
        """
        if x.shape[1] != self.layers_dim[0]:
            raise InternalError(f'Expected n_features to be {self.layers_dim[0]}, got {x.shape[1]}.')
        y_pred = list()
        for i in range(len(x)):
            self.A[0] = x[i].reshape((self.layers_dim[0], 1))
            self._forward_pass()
            y_pred.append(self.A[-1].squeeze())
        return np.array(y_pred)

    def _init_weights(self):
        """Routine to initialize weights and biases
        as per Xavier initialization"""
        for i in range(1, self.n_layers):
            self.W[i] = np.random.RandomState(seed=self.random_state).randn(self.layers_dim[i], self.layers_dim[i - 1])
            self.W[i] *= np.power(1 / self.layers_dim[i - 1], 0.5)
            self.b[i] = np.zeros((self.layers_dim[i], 1))

    def _run(self):
        """Routine to initiate forward prop and backward prop"""
        n_batches = self.x_train.shape[0] // self.batch_size
        for it in tqdm(range(self.max_iter), desc='Epoch'):
            error = 0
            for i in range(n_batches):
                self.A[0] = self.x_train[self.batch_size * i:self.batch_size * (i + 1), :]
                self.A[0] = self.A[0].T
                self._forward_pass()
                y_true = self.y_train[self.batch_size * i:self.batch_size * (i + 1), :]
                y_pred = self.A[-1].T
                error += self._calc_error(y_true, y_pred)
                self._backprop(y_true.T)
                self._update_weights()
            train_acc = self.score(self.x_train, self.y_train)
            val_acc = self.score(self.x_val, self.y_val)
            self.error_curve[it] = error
            self.train_acc[it] = train_acc
            self.val_acc[it] = val_acc

    def _forward_pass(self):
        """Routine for forward propagation"""
        for i in range(1, self.n_layers):
            self.Z[i] = np.dot(self.W[i], self.A[i - 1]) + self.b[i]
            self.A[i] = sigmoid(self.Z[i])

    def _backprop(self,
                  y_true: np.ndarray):
        """Routine for backward propagation
        -----------------------------------
        Input:
        :param y_true: True target labels (one-hot encoded)
        """
        self.dA[-1] = (self.A[-1] - y_true)
        for i in range(self.n_layers - 1, 0, -1):
            self.dZ[i] = self.dA[i] * self.A[i] * (1 - self.A[i])
            self.dW[i] = np.dot(self.dZ[i], self.A[i - 1].T) / self.batch_size
            self.db[i] = np.sum(self.dZ[i], axis=1, keepdims=True) / self.batch_size
            self.dA[i - 1] = np.dot(self.W[i].T, self.dZ[i])

    def _update_weights(self):
        """Routine to update weights and biases"""
        for i in range(1, self.n_layers):
            self.W[i] -= self.learning_rate * self.dW[i]
            self.b[i] -= self.learning_rate * self.db[i]

    def _calc_error(self,
                    y_true: np.ndarray,
                    y_pred: np.ndarray) -> float:
        """Routine to calculate error for a given batch
        -----------------------------------------------
        :param y_true: True target labels (one-hot encoded)
        :param y_pred: Predicted target labels (one-hot encoded)
        """
        return (np.square(y_true - y_pred)).mean(axis=None) / 2

    def _calc_accuracy(self,
                       y_true: np.ndarray,
                       y_pred: np.ndarray) -> float:
        """Routine to return accuracy score for predicted labels
        --------------------------------------------------------
        :param y_true: True labels (one-hot encoded)
        :param y_pred: Predicted labels (one-hot encoded)
        """
        y_true_ = y_true.argmax(axis=1)
        y_pred_ = y_pred.argmax(axis=1)
        return (y_true_ == y_pred_).sum() / len(y_true_)
