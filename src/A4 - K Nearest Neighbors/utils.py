"""Helper function for the module

Author: Utkarsh Patel (18EC35034)
This module is a part of MIES KNN assignment
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

ATTRIBUTES = ['unif_cell_size', 'unif_cell_shape', 'marg_adhesion', 'single_epith_cell_size',
              'bare_nuclei', 'bland_chrom', 'norm_nucleoi', 'mitoses']

LABELS = ['Benign', 'Malignant']

K = [1, 3, 5, 7]

def split_x_y(df: pd.DataFrame) -> tuple:
    """Routine to split dataframe into instances and labels"""
    x = df[ATTRIBUTES].to_numpy()
    y = df['class'].to_numpy().squeeze()
    y = (y == 4)
    return x, y

def plot_features(x_train: np.ndarray,
                  y_train: np.ndarray,
                  x_test: np.ndarray,
                  y_test: np.ndarray):
    """Routine to plot the features for training and test set"""
    if os.path.exists('scatter_plot.png'):
        os.remove('scatter_plot.png')
    plt.figure(figsize=(24, 12))
    mask = np.random.permutation(8)[:2]
    plt.subplot(121)
    scatter = plt.scatter(x_train[:, mask[0]], x_train[:, mask[1]], c=y_train)
    plt.title('Training set')
    plt.xlabel(ATTRIBUTES[mask[0]])
    plt.ylabel(ATTRIBUTES[mask[1]])
    plt.legend(handles=scatter.legend_elements()[0],
               title="Label",
               labels=LABELS)
    plt.subplot(122)
    scatter = plt.scatter(x_test[:, mask[0]], x_test[:, mask[1]], c=y_test)
    plt.title('Test set')
    plt.xlabel(ATTRIBUTES[mask[0]])
    plt.ylabel(ATTRIBUTES[mask[1]])
    plt.legend(handles=scatter.legend_elements()[0],
               title="Label",
               labels=LABELS)
    plt.savefig('scatter_plot.png')
    plt.close()