"""Wrapper for Decision Tree (ID3) algorithm with information gain as splitting criteria

Author: Utkarsh Patel (18EC35034)
This module is a part of MIES coding assignment-1
"""

import numpy as np
import pandas as pd

INFO_GAIN_THRESHOLD = 0
MAX_DEPTH = 10

def entropy(labels: np.ndarray) -> float:
    """Routine to computing entropy for passed labels
    -------------------------------------------------
    Input
    :param labels: list of labels

    Output
    Returns the entropy
    """
    n_labels = len(labels)
    if n_labels <= 1:
        return 0
    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)
    if n_classes <= 1:
        return 0
    ent = 0
    for x in probs:
        ent -= x * np.log2(x)
    return ent

class InternalError(Exception):
    """Exception handling for the module"""
    pass

class Node:
    """Wrapper for Node in the Decision Tree (ID3) algorithm"""
    def __init__(self,
                 x: pd.DataFrame,
                 y: np.ndarray,
                 depth: int = 1,
                 max_depth: int = MAX_DEPTH,
                 info_gain_threshold: float = INFO_GAIN_THRESHOLD):
        """Routine to initialize Node object
        ------------------------------------
        Input
        :param x: instances to fit
        :param y: corresponding labels
        :param depth: current depth of object in the decision tree
        :param max_depth: maximum allowed depth
        :param info_gain_threshold: minimum information gain to make a split
        """
        self.x = x
        self.y = y
        self.children = None
        self.attribute = None
        self.threshold = None
        self.distribution = self.get_distribution()
        self.dominant_label = self.get_dominant_label()
        self.leaf = (len(self.distribution) == 1 or depth >= max_depth)
        self.depth = depth
        self.max_depth = max_depth
        self.info_gain_threshold = info_gain_threshold
        self.name = None
        self.info_gain = 0
        self.entropy = 0
        if not self.leaf:
            self.split()
        else:
            self.name = self.dominant_label

    def __str__(self):
        """Routine to implicitly convert Node object to str object for printing"""
        s = '\t' * (self.depth - 1) + f'data = {self.distribution}\n'
        if self.leaf:
            s += '\t' * (self.depth - 1) + f'class = {self.dominant_label}\n'
        else:
            s += '\t' * (self.depth - 1) + f'info_gain = {self.info_gain}\n'
            s += '\t' * (self.depth - 1) + f'if {self.name}:\n'
            s += self.children[0].__str__()
            s += '\t' * (self.depth - 1) + 'else:\n'
            s += self.children[1].__str__()
        return s

    def get_distribution(self) -> dict:
        """Routine to get the distribution of labels in the given set
        -------------------------------------------------------------
        Output
        Returns a dictionary representing the class distribution
        """
        label_count = dict()
        for label in self.y:
            try:
                label_count[label] += 1
            except KeyError:
                label_count[label] = 1
        return label_count

    def get_dominant_label(self) -> str:
        """Routine to get the dominant label of the set
        -----------------------------------------------
        Output
        Returns the dominant label
        """
        dominant_label = None
        max_count = 0
        for label, count in self.distribution.items():
            if count > max_count:
                max_count = count
                dominant_label = label
        return dominant_label

    def calc_info_gain(self, attribute: str):
        """Routine to calculate information gain for given attribute"""
        thresholds = self.x[attribute].unique().tolist()
        thresholds.sort()
        max_info_gain = 0
        best_threshold = None
        for threshold in thresholds[1:]:
            mask = self.x[attribute] < threshold
            y_left, y_right = self.y[mask], self.y[~mask]
            entropy_left, entropy_right = entropy(y_left), entropy(y_right)
            info_gain = self.entropy - (entropy_left * len(y_left) + entropy_right * len(y_right)) / len(self.y)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_threshold = threshold
        return max_info_gain, best_threshold

    def split(self):
        """Routine to split the training instances using (attribute, threshold) pair with maximum information gain"""
        best_attribute = None
        best_threshold = None
        max_info_gain = 0
        self.entropy = entropy(self.y)
        for attribute in self.x.columns:
            info_gain, threshold = self.calc_info_gain(attribute)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_attribute = attribute
                best_threshold = threshold
        if max_info_gain < self.info_gain_threshold:
            self.leaf = True
            self.name = self.dominant_label
            return
        self.attribute = best_attribute
        self.threshold = best_threshold
        self.info_gain = max_info_gain
        self.name = f'{best_attribute} < {best_threshold}'
        mask = self.x[best_attribute] < best_threshold
        x_left, x_right = self.x[mask], self.x[~mask]
        y_left, y_right = self.y[mask], self.y[~mask]
        self.children = [Node(x_left, y_left, self.depth + 1, self.max_depth, self.info_gain_threshold),
                         Node(x_right, y_right, self.depth + 1, self.max_depth, self.info_gain_threshold)]

class DecisionTree:
    """Wrapper for Decision Tree (ID3) algorithm"""
    def __init__(self,
                 max_depth: int = MAX_DEPTH,
                 info_gain_threshold: float = INFO_GAIN_THRESHOLD):
        """Routine to initialize the DecisionTree object
        ------------------------------------------------
        Input
        :param max_depth: maximum allowed depth
        :param info_gain_threshold: minimum information gain required to split
        """
        self.root = None
        self.max_depth = max_depth
        self.info_gain_threshold = info_gain_threshold

    def fit(self,
            x: pd.DataFrame,
            y: np.ndarray):
        """Routine to build and train the decision tree on training set
        ---------------------------------------------------------------
        Input
        :param x: training instances
        :param y: training labels
        """
        self.root = Node(x, y,
                         max_depth=self.max_depth,
                         info_gain_threshold=self.info_gain_threshold)

    def score(self,
              x: pd.DataFrame,
              y: np.ndarray,
              root: Node = None) -> float:
        """Routine to test the decision tree on test set
        ------------------------------------------------
        Input
        :param x: test instances
        :param y: test labels
        :param root: subtree to be used

        Output
        Returns the accuracy of the prediction
        """
        y_hat = self.predict(x, root)
        return self._score(y, y_hat)

    def predict(self,
                x: pd.DataFrame,
                root: Node = None) -> np.ndarray:
        """Routine to predict the labels of passed instances using passed subtree
        -------------------------------------------------------------------------
        Input
        :param x: validation instances
        :param root: subtree to be used

        Output
        Returns the predicted labels of the instances
        """
        if root is None:
            root = self.root
        labels = []
        for i in range(len(x)):
            labels.append(self._predict(x.iloc[i], root))
        return np.array(labels)

    def prune(self,
              x: pd.DataFrame,
              y: np.ndarray):
        """Routine to prune the decision tree
        -------------------------------------
        :param x: validation instances
        :param y: validation labels
        """
        self._prune(self.root, x, y)

    def __str__(self):
        """Routine to implicitly convert the DecisionTree object to str object for printing"""
        if self.root is None:
            raise InternalError('Attempt to print an empty tree')
        return self.root.__str__()

    @staticmethod
    def _predict(x: pd.Series,
                 node: Node) -> str:
        """Static routine to predict the label of passed instance using passed subtree
        ------------------------------------------------------------------------------
        Input
        :param x: instance to be predicted
        :param node: subtree to be used

        Output
        Returns the predicted label of the instance
        """
        while not node.leaf:
            if x[node.attribute] < node.threshold:
                node = node.children[0]
            else:
                node = node.children[1]
        return node.dominant_label

    @staticmethod
    def _score(y: np.ndarray,
               y_hat: np.ndarray) -> float:
        """Static routine to calculate accuracy of the prediction
        ---------------------------------------------------------
        Input
        :param y: true labels
        :param y_hat: predicted labels

        Output
        Returns the accuracy
        """
        return (y == y_hat).sum() / len(y)

    def _can_prune(self,
                   node: Node,
                   x: pd.DataFrame,
                   y: np.ndarray) -> bool:
        """Routine to check whether passed node can be pruned or not
        ------------------------------------------------------------
        Input
        :param node: node for which the condition is to be checked
        :param x: validation instances for the current node
        :param y: corresponding validation labels

        Output
        Returns True if passed node can be pruned, False otherwise
        """
        if len(np.unique(y)) <= 1:
            return True
        y_hat = np.array([node.dominant_label] * len(y))
        score_w_prune = self._score(y, y_hat)     # score with pruning
        score_wo_prune = self.score(x, y, node)   # score without pruning
        return score_w_prune > score_wo_prune

    def _prune(self,
               node: Node,
               x: pd.DataFrame,
               y: np.ndarray):
        """Recursive pruning utility for the decision tree
        --------------------------------------------------
        Input
        :param node: node to be pruned
        :param x: validation instances
        :param y: validation labels
        """
        if node.leaf:
            return
        if len(np.unique(y)) <= 1:  # if there exists only one class, prune the node to leaf
            node.leaf = True
            node.name = node.dominant_label
            node.children = None
            return
        mask = x[node.attribute] < node.threshold
        self._prune(node.children[0], x[mask], y[mask])
        self._prune(node.children[1], x[~mask], y[~mask])
        if self._can_prune(node, x, y):
            node.leaf = True
            node.name = node.dominant_label
            node.children = None