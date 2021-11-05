"""Wrapper for KNN algorithm

Author: Utkarsh Patel (18EC35034)
This module is a part of MIES KNN assignment
"""

import numpy as np

METRICS_ALLOWED = ['euclidean', 'norm-euclidean', 'cosine']

class InternalError(Exception):
    """Exception handling for the module"""
    pass

class KNN:
    """Wrapper for KNN algorithm"""
    def __init__(self,
                 n_neighbors: int = 5,
                 metric: str = 'euclidean'):
        """Routine to initialize KNN algorithm
        --------------------------------------
        Input:
        :param n_neighbors: Number of neighbors to use by default for queries
        :param metric: Distance metric to be used in the algorithm
        """
        self.n_neighbors = n_neighbors
        self.metric = metric

        self.n_features = 0       # Number of features in a sample
        self.n_samples = 0        # Number of samples
        self.samples = None       # Samples
        self.samples_norm = None  # Normalized samples
        self.labels = None        # Labels

    def fit(self,
            x: np.ndarray,
            y: np.ndarray):
        """Routine to train KNN on sample data
        --------------------------------------
        Input:
        :param x: Training instances
        :param y: Target labels
        """
        self.samples = x
        self.samples_norm = x / np.linalg.norm(x, axis=1, keepdims=True, ord=2)
        self.n_samples = x.shape[0]
        self.n_features = x.shape[1]
        self.labels = y

    def score(self,
              x: np.ndarray,
              y: np.ndarray) -> float:
        """Routine to test KNN
        ----------------------
        Input:
        :param x: Test instances
        :param y: Gold labels
        """
        y_pred = self.predict(x)
        return (y == y_pred).sum() / len(y)

    def predict(self,
                x: np.ndarray) -> np.ndarray:
        """Routine to predict the labels for query samples
        --------------------------------------------------
        Input:
        :param x: Query instances
        """
        if self.n_features != x.shape[1]:
            raise InternalError(f'Expected n_features to be {self.n_features}, got {x.shape[1]}.')
        distance = self._get_distance(x)
        assert distance.shape == (x.shape[0], self.n_samples)
        if self.metric != 'cosine':
            neighbor_ids = np.argpartition(distance, self.n_neighbors)[:, :self.n_neighbors]
        else:
            neighbor_ids = np.argpartition(distance, -self.n_neighbors)[:, -self.n_neighbors:]
        candidate_labels = self.labels[neighbor_ids]
        predicted_labels = np.apply_along_axis(lambda z: np.bincount(z).argmax(), axis=1, arr=candidate_labels)
        return predicted_labels

    def _get_distance(self,
                      x: np.ndarray) -> np.ndarray:
        """Routine to compute distances between training samples and query samples
        --------------------------------------------------------------------------
        Input:
        :param x: Query instances
        """
        if self.metric == 'euclidean':
            return np.sqrt(((self.samples - x[:, np.newaxis])**2).sum(axis=2))
        elif self.metric == 'norm-euclidean':
            u = self.samples - self.samples.mean(axis=1, keepdims=True)
            v = x - x.mean(axis=1, keepdims=True)
            num = 0.5 * ((u - v[:, np.newaxis]) ** 2).sum(axis=2)
            den = np.zeros((x.shape[0], self.n_samples))
            den += np.square(np.linalg.norm(u, axis=1, ord=2).reshape((1, self.n_samples)))
            den += np.square(np.linalg.norm(v, axis=1, ord=2).reshape((x.shape[0], 1)))
            return num / den
        elif self.metric == 'cosine':
            x_ = x / np.linalg.norm(x, axis=1, keepdims=True, ord=2)
            return np.dot(x_, self.samples_norm.T)
        else:
            raise InternalError(f'{self.metric} doesn\'t exist in {METRICS_ALLOWED}')