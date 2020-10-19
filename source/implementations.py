# -*- coding: utf-8 -*-
import numpy as np
import math
from proj1_helpers import *

def standardize(data):
    data -= np.mean(data)
    data /= np.std(data)
    return data

def mae(error):
    return np.mean(np.abs(error))

def compute_mse(y, tx, w):
    """Calculate the loss """
    error = y- tx.dot(w)
    return 0.5*(1/tx.shape[0])*np.mean(error*error)

def sigmoid(t):
    """
    Applies the sigmoid function to a given input t.
    """
    return 1. / (1. + np.exp(-t))

def compute_logistic_loss(y, tx, w):
    """
    Computes the loss as the negative log likelihood of picking the correct label.

    """
    tx_dot_w = tx @ w
    return np.sum(np.log(1. + np.exp(tx_dot_w)) - y * tx_dot_w)

def compute_logistic_gradient(y, tx, w):
    """
    Computes the gradient of the loss function used in logistic regression.
    """
    return tx.T@(sigmoid(tx@w) - y)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using stochastic gradient descent 

    """
    threshold = 1e-8
    ws = [initial_w]
    losses = []
    w = initial_w
    for _ in range(max_iters):
        loss = compute_logistic_loss(y, tx, w)
        gradient = compute_logistic_gradient(y, tx, w)
        w = w - gamma * gradient
        ws.append(w)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break # convergence criterion met
    return ws[-1], losses[-1]

def compute_gradient(y, tx, w):
    '''
    compute the gradient
    '''
    l = tx.shape[0]
    error = y - np.dot(tx, w)
    return -(1/l)*np.dot(tx.T, error)

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    error = y - tx.dot(w)
    grad = -tx.T.dot(error) / len(error)
    return grad, error


def build_poly(X,degree) : 

    # initialize X_transform 
    X_transform = X 
    j = 0
    for j in range( degree + 1 ) : 
        if j > 1 : 
            x_pow = np.power(X, j) 
            # append x_pow to X_transform 
            X_transform = np.append( X_transform, x_pow, axis = 1 ) 
    X=standardize(X)
    return X_transform

def pre_process(tx,method,degree):
    '''Pre-processing of the data by mean or median'''
    tx[tx==-999]=np.nan
    tx = np.delete(tx, [4,6,12,24,25,27,28], axis=1)
    if method == 'mean' or method == 'median':
        
        if method == 'mean':  
            col = np.nanmean(tx,axis = 0) #change median by mean if you want mean
        if method == 'median':
            col = np.nanmedian(tx,axis = 0)
        indices = np.where(np.isnan(tx))
        tx[indices]=np.take(col,indices[1])
    if method == 'drop':
        mask = np.any(np.isnan(tx), axis=1)
        tx = tx[~mask]
        y = y[~mask]
        tx = build_poly(tx,degree)
        return tx,y
    tx = standardize(tx)
    tx = build_poly(tx,degree)
    return tx

def test(y_pred,y_test):
    true= 0
    for i,value in enumerate(y_pred):
        if value == y_test[i]:
            true += 1
    return true/len(y_test)



def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_mse(y,tx,w)
        grad = compute_gradient(y,tx,w)
        w -= gamma*grad
    return w,loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    #http://blog.datumbox.com/tuning-the-learning-rate-in-gradient-descent/
    w = initial_w

    for n_iter in range(max_iters):
        # compute a stochastic gradient and loss
        grad,_ = compute_stoch_gradient(y, tx, w)
        # update w through the stochastic gradient update
        w = w - gamma * grad
        # calculate loss
        loss = compute_mse(y, tx, w)
        # store w and loss
    return w,loss

def least_squares(y,tx):
    """calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w= np.linalg.solve(a, b)
    loss = compute_mse(y,tx,w)
    return w,loss

def ridge_regression(y, tx, lambda_):
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = np.sqrt(compute_mse(y,tx,w))
    return w,loss


def grid_search(lambdas,ratio,degrees,gammas,method,tx,y,regression,verbose = False):
    max_iters = 100
    best_lambda = -1
    best_degree = -1
    best_acc = 0 
    
    for degree in degrees:
        tx_int = pre_process(tx,method,degree)
        x_tr, x_te, y_tr, y_te=split_data(tx_int, y, ratio)
        if regression == 'ls' or regression == 'ridge':    
            for ind, lambda_ in enumerate(lambdas):
                if regression == 'ls':
                    weight,_  = least_squares(y_tr,x_tr)
                    y_pred = predict_labels(weight, x_te)
                    acc = test(y_pred,y_te)
                    if verbose:
                        print('degree = {g},acc = {acc}'.format(g=degree,acc =acc))
                    if acc > best_acc:
                        best_lambda = lambda_
                        best_degree = degree
                        best_acc = acc
                    break
                if regression == 'ridge': 
                    weight,_ = ridge_regression(y_tr, x_tr, lambda_)
                    if verbose:
                        print('degree = {g}, lambda = {l}'.format(g=degree,l=lambda_))
                        
            y_pred = predict_labels(weight, x_te)
            acc = test(y_pred,y_te)
            if verbose:
                print('acc = {acc}'.format(acc =acc))
                print('####################################')
            if acc > best_acc:
                best_lambda = lambda_
                best_degree = degree
                best_acc = acc            
        if regression == 'lsGD' or regression == 'lsSGD' or regression=="logistic":            
            for gamma in gammas:
                if regression == 'lsGD':
                    initial_w = np.ones(x_tr.shape[1])
                    weight,_ = least_squares_GD(y_tr, x_tr, initial_w, max_iters, gamma)
                    if verbose:
                        print('degree = {g}, gamma = {l}'.format(g=degree,l=gamma))
                if regression == 'lsSGD':
                    initial_w = np.ones(x_tr.shape[1])
                    weight,_ = least_squares_SGD(y_tr, x_tr, initial_w, max_iters, gamma)
                    if verbose:
                        print('degree = {g}, gamma = {l}'.format(g=degree,l=gamma))
                if regression =="logistic":
                    initial_w = np.ones(x_tr.shape[1])
                    weight,_= logistic_regression(y_tr, x_tr, initial_w, max_iters, gamma)
                    if verbose:
                        print('degree = {g}, gamma = {l}'.format(g=degree,l=gamma))

                y_pred = predict_labels(weight, x_te)
                acc = test(y_pred,y_te)
                if verbose:
                    print('acc = {acc}'.format(acc =acc))
                    print('####################################')

                if acc > best_acc:
                    best_lambda = gamma
                    best_degree = degree
                    best_acc = acc
        if verbose:
            print('-----------------------------------------')
            print('degree= {degree} ,acc = {acc}'.format(degree=degree,acc =acc))
            print('-----------------------------------------')
    best_model = [best_lambda,best_degree,best_acc]
    print(best_model)
    return best_model
    
def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed random
    np.random.seed(seed)
    
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te
      
  