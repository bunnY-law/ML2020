# -*- coding: utf-8 -*-
import numpy as np

def mse(error):
    return 1/2*np.mean(error**2)

def mae(error):
    return np.mean(np.abs(error))

def compute_mse(y, tx, w):
    """Calculate the loss """
    error = y- tx.dot(w)
    return mse(error)

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    error = y -tx.dot(w)
    grad = -1/len(error)*tx.T.dot(error)
    return grad,error



def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    error = y - tx.dot(w)
    grad = -tx.T.dot(error) / len(error)
    return grad, error

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # TODO: compute gradient and loss
        grad,error = compute_gradient(y,tx,w)
        loss = mse(error)
        # TODO: update w by gradient
        w = w- gamma*grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return ws[-1],losses[-1]

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        # compute a stochastic gradient and loss
        grad, _ = compute_stoch_gradient(y, tx, w)
        # update w through the stochastic gradient update
        w = w - gamma * grad
        # calculate loss
        loss = compute_mse(y, tx, w)
        # store w and loss
        ws.append(w)
        losses.append(loss)
    return ws[-1],losses[-1]

def least_squares(y,tx):
    """calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w= np.linalg.solve(a, b)
    loss = compute_mse(y,tx,w)
    return w,loss

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    aI = lamb * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y,tx,w)
    return w,loss