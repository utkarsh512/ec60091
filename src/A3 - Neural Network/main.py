"""Unittesting Neural Network model

Author: Utkarsh Patel
This module is a part of MIES Coding Assignment #3
"""

from model import *
import pandas as pd
import numpy as np
import argparse
import os
from utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help='path to dataset')
    parser.add_argument('--hidden_layers', nargs='+', type=int, required=True, help='dimension of hidden layers')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for NN')
    parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size used while training')
    parser.add_argument('--max_iter', type=int, default=500, help='maximum iterations to run')
    parser.add_argument('--validation_fraction', type=float, default=0.2, help='fraction to be used as validation')
    parser.add_argument('--test_fraction', type=float, default=0.2, help='fraction to be used in test')
    parser.add_argument('--random_state', type=int, default=42, help='seed for random number generation')
    args = parser.parse_args()

    if os.path.exists('train_scatter_plot.png'):
        os.remove('train_scatter_plot.png')
    if os.path.exists('test_scatter_plot.png'):
        os.remove('test_scatter_plot.png')
    if os.path.exists('loss.png'):
        os.remove('loss.png')
    if os.path.exists('accuracy.png'):
        os.remove('accuracy.png')

    # Reading the dataset
    df = pd.read_csv(args.dir)
    x, y = split_x_y(df)

    np.random.RandomState(seed=args.random_state).shuffle(x)
    np.random.RandomState(seed=args.random_state).shuffle(y)

    # Splitting dataset into training and test set
    n_partition = int(x.shape[0] * args.test_fraction)
    x_train, x_test = x[n_partition:], x[:n_partition]
    y_train, y_test = y[n_partition:], y[:n_partition]

    # Normalization
    mean = x_train.mean(axis=0)
    std_dev = x_train.std(axis=0)
    x_train_norm = (x_train - mean) / std_dev
    x_test_norm = (x_test - mean) / std_dev

    # Scatter plot for Before v/s After normalization
    plot_features_train(x_train, x_train_norm, y_train)

    # Setting up the model
    clf = NeuralNetwork(hidden_layers=args.hidden_layers,
                        learning_rate=args.learning_rate,
                        batch_size=args.batch_size,
                        max_iter=args.max_iter,
                        validation_fraction=args.validation_fraction,
                        random_state=args.random_state)

    # Training and testing
    clf.fit(x_train_norm, y_train)
    acc = clf.score(x_test_norm, y_test)
    y_pred = clf.predict(x_test_norm).argmax(axis=1).squeeze()
    plot_features_test(x_test_norm, y_test, y_pred)
    plot_loss(clf.error_curve)
    plot_acc(clf.train_acc, clf.val_acc)
    print(f'\nMSE Loss: {clf.error_curve[-1]}')
    print(f'Training acc: {clf.train_acc[-1]} - Validation acc: {clf.val_acc[-1]}')
    print(f'Test acc: {acc}')

    # Predicting class labels
    x_query = np.array([[100, 50, 20, 55.5, 42, 23, 35, 11],
                        [110, 74, 25, 153.6, 47.4, 15.5, 11, 10],
                        [106, 73, 16, 70.3, 47.4, 29.9, 33, 19],
                        [94, 81, 20, 132.9, 33.5, 34.2, 38, 10],])
    x_query_norm = (x_query - mean) / std_dev
    y_query = clf.predict(x_query_norm).argmax(axis=1).squeeze()
    labels = list()
    for e in y_query:
        labels.append(decode(e))
    print(f'\nPredicted labels for query vectors:\n')
    print('\tQuery Vector\t\t\t\t\t\tLabel')
    for i in range(len(x_query)):
        print(f'\t{x[i]}\t{labels[i]}')
    print('\nDone, check the directory for the plots.')

if __name__ == '__main__':
    main()





