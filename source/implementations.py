# -*- coding: utf-8 -*-
import numpy as np

def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
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
        loss = compute_loss(y, tx, w)
        # store w and loss
        ws.append(w)
        losses.append(loss)
    return ws[-1],losses[-1]

def least_squares(y,tx):
    #Mean tx and y
    mean_tx = np.mean(tx)
    mean_y = np.mean(y)
    n = len(tx)
    #Calculate the splope and the offset:
    num = 0
    den = 0
    for i in range(n):
        num += (tx[i]-mean_tx)*(y[i]-y_mean)
        den += (tx[i]-mean_tx)**2
    w0 = num/den
    w1 = mean_y -(w0*mean_tx)
    w = [w0,w1]
    loss = compute_loss(y,tx,w)
    return w,loss

def ridge_regression(y,tx,lambda_):
    w = []
    # normal of tx
    A = tx.T @ tx
    # Identity matrix with same shape as A
    I = np.eye(A.shape[0])
    # 
    c = tx.T @ y
    for lambVal in range(1, lambda_+1):
        # Set up equations Bw = c        
        lamb_I = lambVal * I
        B = A + lamb_I
        # Solve for w
        w_i = np.linalg.solve(B,c)
        w.append(w_i)
    loss = compute_loss(y,tx,w)
    return w,loss