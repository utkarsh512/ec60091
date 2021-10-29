"""Helper functions for the program

Author: Utkarsh Patel (18EC35034)
This module is a part of MIES Coding Assignment #3
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ATTRIBUTES = ['length', 'diameter', 'height', 'whole-weight',
              'shucked-weight', 'viscera-weight',
              'shell-weight', 'rings']

LABEL_ENCODING = {'M': 0,
                  'F': 1,
                  'I': 2}

LABEL_DECODING = ['Male', 'Female', 'Both']

def encode(lbl):
    return LABEL_ENCODING[lbl]

def decode(lbl):
    return LABEL_DECODING[lbl]

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Routine to calculate sigmoid activation"""
    return 1 / (1 + np.exp(-x))

def split_x_y(df: pd.DataFrame) -> tuple:
    """Routine to split dataframe into instances and labels"""
    x = df[ATTRIBUTES].to_numpy()
    df['sex'] = df['sex'].apply(lambda z: encode(z))
    foo = df[['sex']].to_numpy().squeeze()
    y = np.zeros((foo.size, 3))
    y[np.arange(foo.size), foo] = 1
    return x, y

def plot_features_train(x_u: np.ndarray,
                        x: np.ndarray,
                        y: np.ndarray):
    """Routine to plot the features before and after normalization"""
    plt.figure(figsize=(24, 12))
    mask = np.random.permutation(8)[:2]
    y_ = y.argmax(axis=1)
    plt.subplot(121)
    scatter = plt.scatter(x_u[:, mask[0]], x_u[:, mask[1]], c=y_)
    plt.title('Before normalization')
    plt.xlabel(ATTRIBUTES[mask[0]])
    plt.ylabel(ATTRIBUTES[mask[1]])
    plt.legend(handles=scatter.legend_elements()[0],
               title="Gender",
               labels=LABEL_DECODING)
    plt.subplot(122)
    scatter = plt.scatter(x[:, mask[0]], x[:, mask[1]], c=y_)
    plt.title('After normalization')
    plt.xlabel(ATTRIBUTES[mask[0]])
    plt.ylabel(ATTRIBUTES[mask[1]])
    plt.legend(handles=scatter.legend_elements()[0],
               title="Gender",
               labels=LABEL_DECODING)
    plt.savefig('train_scatter_plot.png')
    plt.close()

def plot_features_test(x: np.ndarray,
                       y_true: np.ndarray,
                       y_pred: np.ndarray):
    """Routine to plot the features for test data"""
    plt.figure(figsize=(24, 12))
    mask = np.random.permutation(8)[:2]
    y_ = y_true.argmax(axis=1)
    plt.subplot(121)
    scatter = plt.scatter(x[:, mask[0]], x[:, mask[1]], c=y_)
    plt.title('True label')
    plt.xlabel(ATTRIBUTES[mask[0]])
    plt.ylabel(ATTRIBUTES[mask[1]])
    plt.legend(handles=scatter.legend_elements()[0],
               title="Gender",
               labels=LABEL_DECODING)
    plt.subplot(122)
    scatter = plt.scatter(x[:, mask[0]], x[:, mask[1]], c=y_pred)
    plt.title('Predicted label')
    plt.xlabel(ATTRIBUTES[mask[0]])
    plt.ylabel(ATTRIBUTES[mask[1]])
    plt.legend(handles=scatter.legend_elements()[0],
               title="Gender",
               labels=LABEL_DECODING)
    plt.savefig('test_scatter_plot.png')
    plt.close()

def plot_loss(data: list):
    """Routine to plot the loss for each iteration"""
    plt.switch_backend('Agg')
    plt.figure(figsize=(24, 12))
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.plot(data)
    plt.title('MSE v/s Iteration')
    plt.savefig('loss.png')
    plt.close()

def plot_acc(train_acc: list,
             val_acc: list):
    """Routine to plot the accuracies for each iteration"""
    plt.switch_backend('Agg')
    plt.figure(figsize=(24, 12))
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.plot(train_acc, label='training')
    plt.plot(val_acc, label='validation')
    plt.title('Accuracy v/s Iteration')
    plt.legend()
    plt.savefig('accuracy.png')
    plt.close()

