# -*- coding: utf-8 -*-
import numpy as np
import math
from proj1_helpers import *


##---------------------------------------------------------------------------
##--------GENERAL----------------------------------------


def mae(error):
    return np.mean(np.abs(error))

def compute_mse(y, tx, w):
    """Calculate the loss """
    e = y- tx@w
    return 1./(2*len(y))*e.dot(e)





##--------------------------------------------------------------------------
##-----------------------METHODS------------------------------------------


##--------------GRADIENT METHODS------------------------------

def compute_gradient(y, tx, w):
    e = y - np.dot(tx, w)
    return -(1./len(y))*tx.T@e

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
   
    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_mse(y,tx,w)
        grad = compute_gradient(y,tx,w)
        w -= gamma*grad
    return w,loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    w = initial_w

    for n_iter in range(max_iters):
        # compute a stochastic gradient and loss
        grad = compute_gradient(y, tx, w)
        # update w through the stochastic gradient update
        w = w - gamma * grad
        # calculate loss
        loss = compute_mse(y, tx, w)
        # store w and loss
    return w,loss





##-----METHODS  USING NORMAL EQUATION---------------------------------------
def least_squares(y,tx):
    """calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w= np.linalg.lstsq(a, b,rcond=None)[0]
    loss = compute_mse(y,tx,w)
    return w,loss

def ridge_regression(y, tx, lambda_):

    E=np.diag(np.concatenate((np.array([[0]]),np.ones((1,np.shape(tx)[1]-1))),1)[0,:])
    
    w=np.linalg.solve(tx.T@tx+lambda_*E,tx.T@y)
    loss=compute_mse(y,tx,w)
    
    return w,loss


##---LOGISTIC REGRESSION METHOD ---------------------------------------------

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def compute_logistic_loss(y, tx, w):
    tx_dot_w = tx @ w
    return np.sum(np.log(1. + np.exp(tx_dot_w)) - y @ tx_dot_w)

def compute_logistic_gradient(y, tx, w):
    return tx.T@(sigmoid(tx@w) - y)/y.shape[0]

def reg_logistic_gradient(y, tx, lambda_, w):
    """return the loss and gradient."""
    loss = compute_logistic_loss(y, tx, w) + lambda_ * np.squeeze(w.T@w)
    gradient = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
    return loss, gradient

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
            break # convergence 
    return ws[-1], losses[-1]
def reg_logistic_regression(y, tx, initial_w,lambda_, max_iters, gamma):
    """
    Logistic regression using stochastic gradient descent 

    """
    threshold = 1e-8
    ws = [initial_w]
    losses = []
    w = initial_w
    for _ in range(max_iters):
        loss,gradient = reg_logistic_gradient(y, tx, lambda_, w)
        w = w - gamma * gradient
        ws.append(w)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break # convergence criterion met
    return ws[-1], losses[-1]






##-------------------------------------------------------------------------
##------------------FEATURE EXPANSION-------------------------------------------------



def build_poly(x, degree, cross_term='true'):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
    n=np.shape(x)[1]
    if cross_term=='true':
        phi=np.zeros((len(x),n*degree+1+np.int(n*(n-1)/2.)))
    else:
        phi=np.zeros((len(x),n*degree+1))

    phi[:,0]=np.ones(len(x))[:]
    k=1
    for d in range(n): 
        for deg in range(1,degree+1):
            phi[:,k]=np.power(x[:,d],deg)
            k=k+1
    if cross_term=='true':
        for i in range(0,n):
            for j in range(i+1,n):
                phi[:,k]=x[:,i]*x[:,j]
                k=k+1
               
    return phi

def cross_term(x, x_0):
    for col1 in range(x_0.shape[1]):
        for col2 in np.arange(col1 + 1, x_0.shape[1]):
            if col1 != col2:
                x = np.c_[x, x_0[:, col1] * x_0[:, col2]]
    return x


def log_term(x, x_0):

    for col in range(0,x_0.shape[1]):
        current_col = x_0[:, col]
        current_col[current_col <= 0] = -np.log(1.-current_col[current_col<=0])
        current_col[current_col > 0] = np.log(1.+current_col[current_col>0])
        x = np.c_[x, current_col]
        
    return x






##-------------------------------------------------------------------------
##-----------PREPROCESS DATA-----------------------------------------------




def remove_outliers(tx):
    median=np.nanmedian(tx,axis = 0)
    for i in range(tx.shape[1]):
            mean = np.mean(tx[:, i])
            std = np.std(tx[:, i])
            #Replace values that are bigger than mean + 3std or smaller than mean - 3std
            tx[:, i][tx[:, i] > mean + 3*std] = median[i]
            tx[:, i][tx[:, i] < mean - 3*std] = median[i]
    return tx

def standardize(data):
    data -= np.mean(data)
    data /= np.std(data)
    return data    



def pre_process(tx,method,degree):
    '''Pre-processing of the data by mean or median'''
    tx[tx==-999]=np.nan
    tx = np.delete(tx, [4,6,12,24,25,27,28], axis=1)
    if method == 'mean' or method == 'median':
        
        if method == 'mean':  
            col = np.nanmean(tx,axis = 0) 
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
    tx = remove_outliers(tx)
    tx = standardize(tx)
    x_0 = tx
    #tx = cross_term(tx,x_0) !Dont need anymore just set last arg of build_poly to true!
    #!!!watch out to apply log_term after build_poly and using x_0 unprocessed data (out 1 columns)
    tx = build_poly(tx,degree,'true')
    tx = log_term(tx,x_0)
    return tx






##----------------------------------------------------------------------------
##----------------------FINAL RUN and TEST------------------------------------




def test(y_pred,y_test):
    true= 0
    for i,value in enumerate(y_pred):
        if value == y_test[i]:
            true += 1
    return true/len(y_test)

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


def grid_search(lambdas,ratio,degrees,gammas,method,tx,y,regression,verbose = False):
    max_iters = 1000
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
                        
                if regression =="reg_logistic":
                    initial_w = np.ones(x_tr.shape[1])
                    weight,_= reg_logistic_regression(y, tx, initial_w,0.00001, max_iters, gamma)
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
    


##---------------OR WITH CROSS CORRELATION----------------------------




def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)



def cross_validation(y, x, k_fold, lambda_, degree, seed=1, method="ridge"):

    """build k indices for k-fold."""
    k_indices=build_k_indices(y, k_fold, seed)
    
    """return the loss of ridge regression."""
    # ***************************************************
    # form data with polynomial degree: TODO
    # ***************************************************   
    phi=pre_process(x,'mean',degree) 
    
    # ***************************************************
    # get k'th subgroup in test, others in train: TODO
    # ***************************************************
    Losses_tr=[]
    Losses_te=[]
    for k in range(0,k_fold):

        l=0
        for i in range(0,len(k_indices)):
            if i!=k:
                if l==0:
                    l=1
                    x_tr=phi[k_indices[i][:]]
                    y_tr=y[k_indices[i][:]]
                else:
                    x_tr=np.concatenate((x_tr,phi[k_indices[i][:]]),0)
                    y_tr=np.concatenate((y_tr,y[k_indices[i][:]]),0)
            else:
                x_te=phi[k_indices[k][:]]
                y_te=y[k_indices[k][:]]

         # ***************************************************# ***************************************************
          # ridge regression
          # ***************************************************
    
        w,mse_tr=ridge_regression(y_tr,x_tr,lambda_)
        mse_te=compute_mse(y_te,x_te,w)
        Losses_tr += [mse_tr]
        Losses_te += [mse_te]
            
    loss_tr = np.mean(Losses_tr)
    loss_te = np.mean(Losses_te)
    var_tr = np.std(Losses_tr)
    var_te = np.std(Losses_te)

    return loss_tr,loss_te, var_tr,var_te

    


    
      
  