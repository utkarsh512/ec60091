"""Wrapper for k-means algorithm for clustering

Author: Utkarsh Patel (18EC35034)
This module is a part of MIES coding assignment #2"""

import numpy as np
from tqdm import tqdm

class InternalError(Exception):
    """Exception handing for the module"""
    pass

class KMeans:
    """Wrapper for k-means clustering"""
    def __init__(self,
                 n_clusters: int = 3,
                 init: str = 'random',
                 max_iter: int = 10,
                 random_state: int = 0):
        """Routine to initialize the KMeans clustering algorithm
        --------------------------------------------------------
        Input:
        :param n_clusters: The number of clusters to form
        :param init: Method for initialization, can be one of {'random', 'k-means++'}
        :param max_iter: Number of time the k-means algorithm will run
        :param random_state: Random state for clustering
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.random_state = random_state

        self.n_features = None       # Number of features in a sample
        self.n_samples = None        # Number of samples
        self.samples = None          # Samples
        self.labels = None           # Labels
        self.cluster_centers = None  # array-like object with shape (n_clusters, n_features)

    def fit(self,
            x: np.ndarray):
        """Routine to train the k-means algorithm on sample data
        --------------------------------------------------------
        Input:
        :param x: Samples for training the algorithm
        """
        self.samples = x
        self.n_samples = x.shape[0]
        self.n_features = x.shape[1]
        self.labels = [None] * x.shape[0]
        self.cluster_centers = self._init_centers()
        for _ in tqdm(range(self.max_iter), desc='iterations'):
            self._run()
        return self

    def fit_predict(self,
                    x: np.ndarray) -> np.ndarray:
        """Routine to train the k-means algorithm on sample data and predict their clusters
        -----------------------------------------------------------------------------------
        Input:
        :param x: Samples for training the algorithm

        Output:
        Returns the cluster labels
        """
        return self.fit(x).labels

    def predict(self,
                x: np.ndarray) -> np.ndarray:
        """Routine to get cluster labels for unseen samples
        ---------------------------------------------------
        Input:
        :param x: Samples for cluster labelling

        Ouput:
        Returns the cluster labels
        """
        if self.n_features != x.shape[1]:
            raise InternalError(f'Expected n_features to be {self.n_features}, got {x.shape[1]}.')
        return np.sqrt(((x - self.cluster_centers[:, np.newaxis])**2).sum(axis=2)).argmin(axis=0)

    def _init_centers(self) -> np.ndarray:
        """Routine to initialize the cluster centers"""
        if self.init == 'random':
            mask = np.random.RandomState(seed=self.random_state).permutation(self.n_samples)[:self.n_clusters]
            return self.samples[mask]
        elif self.init == 'k-means++':
            mask = np.random.RandomState(seed=self.random_state).randint(self.n_samples)
            clusters = np.array([self.samples[mask]])
            while len(clusters) < self.n_clusters:
                distances = np.sqrt(((self.samples - clusters[:, np.newaxis])**2).sum(axis=2))
                idx = distances.min(axis=0).argmax()
                clusters = np.append(clusters, [self.samples[idx]], axis=0)
            return clusters
        else:
            raise InternalError(f'\'{self.init}\' is not a valid initialization method.')

    def _run(self):
        """Routine to carry-out one iteration of k-means clustering"""
        self.labels = np.sqrt(((self.samples - self.cluster_centers[:, np.newaxis])**2).sum(axis=2)).argmin(axis=0)
        new_centers = np.zeros((self.n_clusters, self.n_features))
        label_count = [0] * self.n_clusters
        for i in range(self.n_samples):
            new_centers[self.labels[i]] += self.samples[i]
            label_count[self.labels[i]] += 1
        for i in range(self.n_clusters):
            try:
                new_centers[i] /= label_count[i]
            except ZeroDivisionError:
                pass
        self.cluster_centers = new_centers
