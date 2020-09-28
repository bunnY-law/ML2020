# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np


def load_data(path,sub_sample=False):
    """Load data and convert it to the metrics system."""
    
    x = np.genfromtxt(path, delimiter=",", skip_header=1)
    y = np.genfromtxt(path, delimiter=",", skip_header=1, dtype = str, usecols=1)
    ids = x[:,0].astype(np.int)
    data_input = x[:,2:]
    #need to convert class labels to binary (0,1)
    y_binary = np.ones(len(y))
    y_binary[np.where(y=='b')] = 0
    # sub-sample
    if sub_sample:
        y_binary = y_binary [::50]
        data_input = data_input[::50]
        ids = ids[::50]

    return y_binary, data_input, ids





def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
