"""Training and testing the Decision Tree (ID3) algorithm

Author: Utkarsh Patel (18EC35034)
This module is a part of MIES coding assignment-1
"""

from model import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm

def split_x_y(df: pd.DataFrame) -> tuple:
    """Routine to split dataframe into instances and labels"""
    x = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = df[['class']].to_numpy().squeeze()
    return x, y

def plot(train_acc: list[float],
         val_acc: list[float],
         title: str,
         save_as: str,
         xlabel: str = 'maximum depth',
         ylabel: str = 'accuracy'):
    """Routine to plot the accuracies"""
    plt.switch_backend('Agg')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(train_acc, label='training')
    plt.plot(val_acc, label='validation')
    plt.title(title)
    plt.legend()
    plt.savefig(save_as)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=True, help='path to training set')
    parser.add_argument('--test_dir', type=str, required=True, help='path to test data')
    parser.add_argument('--val_prop', type=float, default=0.3, help='validation proportion')
    args = parser.parse_args()
    line = '-' * 100

    traindf = pd.read_csv(args.train_dir)  # training set
    testdf = pd.read_csv(args.test_dir)    # test set
    traindf = traindf.iloc[np.random.RandomState(seed=42).permutation(len(traindf))]  # Shuffling the training set
    validation_size = int(args.val_prop * len(traindf))
    valdf = traindf[:validation_size]
    traindf = traindf[validation_size:]

    x_train, y_train = split_x_y(traindf)
    x_val, y_val = split_x_y(valdf)
    x_test, y_test = split_x_y(testdf)

    max_depths = range(11)                                # for grid search over maximum allowed depth
    info_gain_thresholds = [0.05 * x for x in range(20)]  # for grid search over information gain threshold

    trees_wo_prune = [None] * len(max_depths)      # list of best unpruned trees for given maximum depth
    train_acc_wo_prune = [None] * len(max_depths)  # training accuracy observed for the best unpruned trees
    val_acc_wo_prune = [None] * len(max_depths)    # validation accuracy observed for the best unpruned trees

    trees_prune = [None] * len(max_depths)         # list of best pruned trees for given maximum depth
    train_acc_prune = [None] * len(max_depths)     # training accuracy observed for the best pruned trees
    val_acc_prune = [None] * len(max_depths)       # validation accuracy observed for the best pruned trees

    print('Building decision trees..')
    with tqdm(total=len(max_depths) * len(info_gain_thresholds)) as pbar:  # grid search for finding optimal hyperparameters
        for i in range(len(max_depths)):
            best_tree = None
            max_val_acc = 0
            train_acc = 0
            for j in range(len(info_gain_thresholds)):
                tree = DecisionTree(max_depth=max_depths[i],
                                    info_gain_threshold=info_gain_thresholds[j])
                tree.fit(x_train, y_train)
                cur_train_acc = tree.score(x_train, y_train)
                cur_val_acc = tree.score(x_val, y_val)
                if cur_val_acc > max_val_acc:
                    best_tree = tree
                    max_val_acc = cur_val_acc
                    train_acc = cur_train_acc
                pbar.update(1)
            trees_wo_prune[i] = best_tree
            train_acc_wo_prune[i] = train_acc
            val_acc_wo_prune[i] = max_val_acc

    print('Pruning decision trees..')
    with tqdm(total=len(max_depths) * len(info_gain_thresholds)) as pbar:  # grid search for finding optimal hyperparameters
        for i in range(len(max_depths)):
            best_tree = None
            max_val_acc = 0
            train_acc = 0
            for j in range(len(info_gain_thresholds)):
                tree = DecisionTree(max_depth=max_depths[i],
                                    info_gain_threshold=info_gain_thresholds[j])
                tree.fit(x_train, y_train)
                tree.prune(x_val, y_val)
                cur_train_acc = tree.score(x_train, y_train)
                cur_val_acc = tree.score(x_val, y_val)
                if cur_val_acc > max_val_acc:
                    best_tree = tree
                    max_val_acc = cur_val_acc
                    train_acc = cur_train_acc
                pbar.update(1)
            trees_prune[i] = best_tree
            train_acc_prune[i] = train_acc
            val_acc_prune[i] = max_val_acc

    if os.path.exists('accuracy_without_pruning.png'):
        os.remove('accuracy_without_pruning.png')
    if os.path.exists('accuracy_with_pruning.png'):
        os.remove('accuracy_with_pruning.png')
    plot(train_acc_wo_prune,
         val_acc_wo_prune,
         title='Accuracy of Decision Tree (ID3) without pruning',
         save_as='accuracy_without_pruning.png')
    plot(train_acc_prune,
         val_acc_prune,
         title='Accuracy of Decision Tree (ID3) with pruning',
         save_as='accuracy_with_pruning.png')

    print(f'\n{line}\nBest unpruned tree..')
    best_unpruned_tree = None
    max_val_acc_unpruned = 0
    for i in range(len(max_depths)):
        if val_acc_wo_prune[i] > max_val_acc_unpruned:
            max_val_acc_unpruned = val_acc_wo_prune[i]
            best_unpruned_tree = trees_wo_prune[i]
    print(f'\n{best_unpruned_tree}\n')
    print(f'training accuracy = {best_unpruned_tree.score(x_train, y_train)}')
    print(f'validation accuracy = {best_unpruned_tree.score(x_val, y_val)}')
    print(f'test accuracy = {best_unpruned_tree.score(x_test, y_test)}')

    print(f'\n{line}\nBest pruned tree..')
    best_pruned_tree = None
    max_val_acc_pruned = 0
    for i in range(len(max_depths)):
        if val_acc_prune[i] > max_val_acc_pruned:
            max_val_acc_pruned = val_acc_prune[i]
            best_pruned_tree = trees_prune[i]
    print(f'\n{best_pruned_tree}\n')
    print(f'training accuracy = {best_pruned_tree.score(x_train, y_train)}')
    print(f'validation accuracy = {best_pruned_tree.score(x_val, y_val)}')
    print(f'test accuracy = {best_pruned_tree.score(x_test, y_test)}')
    print(f'\n{line}\nView \'accuracy_without_pruning.png\' and \'accuracy_with_pruning.png\' to see the performance of all the decision trees')
    print('\nDONE.')

if __name__ == '__main__':
    main()


