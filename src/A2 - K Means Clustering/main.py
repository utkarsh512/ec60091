"""Unittest for k-means clustering algorithm

Author: Utkarsh Patel
This module is a part of MIES coding assignment #2
"""

from model import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

ATTRIBUTES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
LABELS = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

def split_x_y(df: pd.DataFrame) -> tuple:
    """Routine to split dataframe into instances and labels"""
    x = df[ATTRIBUTES].to_numpy()

    encoded_label = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2,
    }

    def encode(lbl):
        return encoded_label[lbl]

    df['label'] = df['label'].apply(lambda z: encode(z))
    y = df[['label']].to_numpy().squeeze()
    return x, y

def correction(y: np.ndarray,
               y_pred: np.ndarray) -> tuple:
    """Routine to apply correction on predicted labels using true lables for correct color mapping
    ----------------------------------------------------------------------------------------------
    Input:
    :param y: True labels
    :param y_pred: Predicted labels

    Output:
    Returns predicted labels with correct color mapping and the mapping itself
    """
    dis = np.zeros((3, 3))
    for i in range(len(y)):
        dis[y_pred[i]][y[i]] += 1
    correct_map = dis.argmax(axis=1)
    if len(np.unique(correct_map)) != 3:
        raise InternalError('Poor convergence obtained.')
    foo = lambda x: correct_map[x]
    return foo(y_pred), correct_map

def jaccard_index(y: np.ndarray,
                  y_pred: np.ndarray) -> list:
    """Routine to compute the Jaccard index after clustering
    --------------------------------------------------------
    Input:
    :param y: True labels
    :param y_pred: Preidcted labels

    Output:
    Returns jaccard score as dictionary
    """
    idx = {
        'label_0_true': [],
        'label_0_pred': [],
        'label_1_true': [],
        'label_1_pred': [],
        'label_2_true': [],
        'label_2_pred': [],
    }
    for i in range(len(y)):
        idx[f'label_{y[i]}_true'].append(i)
        idx[f'label_{y_pred[i]}_pred'].append(i)
    jaccard_score = [0, 0, 0]
    for i in range(3):
        a = set(idx[f'label_{i}_true'])
        b = set(idx[f'label_{i}_pred'])
        jaccard_score[i] = len(a & b) / len(a | b)
    return jaccard_score

def plot(x: np.ndarray,
         y: np.ndarray,
         title: str,
         saveas: str):
    """Routine to plot the clusters
    -------------------------------
    Input:
    :param x: feature vectors
    :param y: cluster labels
    :param title: title of the plot
    :param saveas: name of plot for saving on disk
    """
    masks = []
    for i in range(4):
        for j in range(i + 1, 4):
            masks.append((i, j))
    plt.switch_backend('Agg')
    plt.figure(figsize=(18, 12))
    for i in range(6):
        subplot_ = 230 + i + 1
        plt.subplot(subplot_)
        f1, f2 = masks[i]
        plt.scatter(x[:, f1], x[:, f2], c=y)
        plt.title(title)
        plt.xlabel(ATTRIBUTES[f1])
        plt.ylabel(ATTRIBUTES[f2])
    plt.savefig(saveas)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help='path to the csv file')
    parser.add_argument('--n_clusters', type=int, default=3, help='number of clusters for k-means')
    parser.add_argument('--max_iter', type=int, default=10, help='maximum iteration for k-means to run')
    parser.add_argument('--random_state', type=int, default=0, help='random state for rng')
    args = parser.parse_args()

    # Reading the dataset and plotting the features using true label as a scatter plot
    df = pd.read_csv(args.dir)
    x, y = split_x_y(df)
    if os.path.exists('clustering_using_true_labels.png'):
        os.remove('clustering_using_true_labels.png')
    plot(x, y,
         title='Clustering using True Labels',
         saveas='clustering_using_true_labels.png')

    # Using random initialization in k-means clustering
    print('Clustering using k-means with random initialization')
    clf = KMeans(n_clusters=args.n_clusters,
                 init='random',
                 max_iter=args.max_iter,
                 random_state=args.random_state)
    y_pred = clf.fit_predict(x)
    y_pred, mapping = correction(y=y, y_pred=y_pred)
    cluster_centers_ = clf.cluster_centers
    cluster_centers = [None] * 3
    for i in range(3):
        cluster_centers[mapping[i]] = cluster_centers_[i]
    print('The Jaccard indices and Cluster Centers after k-means clustering with random initializations are:\n')
    jaccard_score = jaccard_index(y, y_pred)
    print(f'\tClusters\tJaccard Index\tCluster Center')
    for i in range(3):
        print(f'\t{LABELS[i]}\t{round(jaccard_score[i], 2)}\t\t{cluster_centers[i]}')
    if os.path.exists('clustering_using_predicted_labels_random_init.png'):
        os.remove('clustering_using_predicted_labels_random_init.png')
    plot(x, y_pred,
         title='Clustering using Predicted Labels (Random Initialization)',
         saveas='clustering_using_predicted_labels_random_init.png')

    # Using k-means++ for clustering
    print('\nClustering using k-means++')
    clf = KMeans(n_clusters=args.n_clusters,
                 init='k-means++',
                 max_iter=args.max_iter,
                 random_state=args.random_state)
    y_pred = clf.fit_predict(x)
    y_pred, mapping = correction(y=y, y_pred=y_pred)
    cluster_centers_ = clf.cluster_centers
    cluster_centers = [None] * 3
    for i in range(3):
        cluster_centers[mapping[i]] = cluster_centers_[i]
    print('\nThe Jaccard indices and Cluster Centers after k-means++ clustering are:\n')
    jaccard_score = jaccard_index(y, y_pred)
    print(f'\tClusters\tJaccard Index\tCluster Center')
    for i in range(3):
        print(f'\t{LABELS[i]}\t{round(jaccard_score[i], 2)}\t\t{cluster_centers[i]}')
    if os.path.exists('clustering_using_predicted_labels_kmeans++.png'):
        os.remove('clustering_using_predicted_labels_kmeans++.png')
    plot(x, y_pred,
         title='Clustering using Predicted Labels (K-Means++)',
         saveas='clustering_using_predicted_labels_kmeans++.png')
    print(f'\nLabels of coordinates of centers are given as {ATTRIBUTES}')
    print('Kindly view the plots generated in parent directory for cluster visualization.\n\nDONE.')

if __name__ == '__main__':
    main()
