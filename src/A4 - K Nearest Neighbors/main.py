"""Unit-testing KNN on Cancer dataset

Author: Utkarsh Patel (18EC35034)
This module is a part of MIES KNN assignment
"""

import pandas as pd
import numpy as np
from model import *
from utils import *

np.seterr(invalid='ignore')

def main():
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    x_train, y_train = split_x_y(df_train)
    x_test, y_test = split_x_y(df_test)
    plot_features(x_train, y_train, x_test, y_test)

    best_hyperparam = (None, None)
    best_score = 0
    print('Processing...\n')
    print('\tMetric              \tNeighbors\tTrain accuracy\tTest accuracy')
    for i in range(len(METRICS_ALLOWED)):
        for j in range(len(K)):
            clf = KNN(n_neighbors=K[j], metric=METRICS_ALLOWED[i])
            clf.fit(x_train, y_train)
            train_acc = clf.score(x_train, y_train)
            test_acc = clf.score(x_test, y_test)
            if test_acc > best_score:
                best_score = test_acc
                best_hyperparam = (i, j)
            print(f'\t{METRICS_ALLOWED[i]:20}\t{K[j]}\t\t{train_acc:.3}\t\t{test_acc:.3}')

    print(f'\nMaximum test accuracy {best_score:.3} achieved using {K[best_hyperparam[1]]} nearest neighbor(s) for {METRICS_ALLOWED[best_hyperparam[0]]} metric.')

if __name__ == '__main__':
    main()