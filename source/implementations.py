# -*- coding: utf-8 -*-
import numpy as np
import math
from proj1_helpers import *
import matplotlib.pyplot as plt




##---------------------------------------------------------------------------
##--------GENERAL----------------------------------------


def mae(error):
    return np.mean(np.abs(error))

def compute_mse(y, tx, w):
    """Calculate the loss """
    e = y- tx@w
    return 1./(2.*len(y))*e.dot(e)


def err_percent(y_test, tx_test, w_trained):
    """return % of wrong predictions when w_trained is applied on test data"""
    y_pred = tx_test@w_trained
    y_pred[y_pred>0] = 1
    y_pred[y_pred<0] = -1
     
    return len(y_pred[y_pred != y_test])/len(y_test) *100.   



##--------------------------------------------------------------------------
##-----------------------METHODS------------------------------------------


##--------------GRADIENT METHODS------------------------------

def compute_gradient(y, tx, w):
    e = y - np.dot(tx, w)
    return -np.sqrt((1./len(y))*tx.T@e)


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
    w= np.linalg.lstsq(a, b)[0]
    loss = compute_mse(y,tx,w)
    return w,loss



def ridge_regression(y, tx, lambda_):

    #E=np.diag(np.concatenate((np.array([[0]]),np.ones((1,np.shape(tx)[1]-1))),1)[0,:])
    E=np.diag(np.ones(np.shape(tx)[1]))
    if lambda_<1e-5:
        w=np.linalg.lstsq(tx.T@tx+lambda_*E,tx.T@y)[0]
    else:
        w=np.linalg.solve(tx.T@tx + lambda_*E,tx.T@y)

    percent_err=err_percent(y,tx,w)   #compute_mse(y,tx,w)
    
    return w, percent_err


##---LOGISTIC REGRESSION METHOD ---------------------------------------------

def sigmoid(x):
    return 1. / (1. + np.exp(-x))



def compute_logistic_loss(y, tx, w):
    tx_dot_w = tx @ w
    return np.sum(np.log(1. + np.exp(tx_dot_w)) - y @ tx_dot_w)



def compute_logistic_gradient(y, tx, w):
    return tx.T@(sigmoid(tx@w) - y)/y.shape[0]



def reg_logistic_gradient(y, tx, w, lambda_):
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



def build_poly(x, degree, cross_term='false'):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    """ also optionly adds cross term multiplications of degree 1"""
    
    n=np.shape(x)[1] #number of features
    
    #for speed, pre initialize ouput matrix at right size
    if cross_term=='true':
        phi=np.zeros((len(x), n*degree+np.int(n*(n-1)/2.)))
    else:
        phi=np.zeros((len(x), n*degree))

    #phi[:,0]=np.ones(len(x))[:] #no need for ones column because we will standardize data
    
    k=0

    #add polynomial terms
    for d in range(n): 
        for deg in range(1,degree+1):
            phi[:,k] = np.power(x[:,d], deg)
            k=k+1

    #add cross terms         
    if cross_term=='true':
        for i in range(0,n):
            for j in range(i+1,n):
                phi[:,k]=x[:,i]*x[:,j]
                k=k+1
               
    return phi



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



def pre_process(tx, method, degree, cross='false', log='false'):
    '''Pre-processing of the data by mean or median'''

    tx[tx==-999]=np.nan
    tx = np.delete(tx, [4,6,12,24,25,27,28], axis=1)

        
    if method == 'mean':  
        col = np.nanmean(tx,axis = 0) 

    if method == 'median':
        col = np.nanmedian(tx,axis = 0)
    indices = np.where(np.isnan(tx))
    tx[indices]=np.take(col,indices[1])

    tx = remove_outliers(tx)
    
    if log == 'true':
        x_0 = tx #save current state of tx so we 
                #can apply log terms only to initial data
    
    #!!!watch out to apply log_term after build_poly and using x_0 unprocessed data (out 1 columns)
    tx = build_poly(tx,degree,cross)
    
    if log=='true':
        tx = log_term(tx,x_0)

    tx = standardize(tx) #standardize only after all expansions for better results 
                         #solving Normal equations 
    return tx






##----------------------------------------------------------------------------
##---------------------- SEARCH FOR BEST METHOD AND HYPERPARAMETERS------------------------------------



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





##--------------------------------------------------------------------------------
##--------------- CROSS VALIDATION----------------------------


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)



def cross_validation(y, x, k_fold, lambda_, degree, seed=1, method="ridge",cross='false',log='false'):

    """build k indices for k-fold."""
    k_indices=build_k_indices(y, k_fold, seed)
    
    """return the loss of ridge regression."""
    # ***************************************************
    # form data with polynomial degree: TODO
    # ***************************************************   
    phi=pre_process(x,'mean',degree,cross,log) 
    
    # ***************************************************
    # perform cross validation
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
    
        w,err_tr=ridge_regression(y_tr,x_tr,lambda_)
        err_te=err_percent(y_te,x_te,w)  
        Losses_tr += [err_tr]
        Losses_te += [err_te]
            
    loss_tr = np.mean(Losses_tr)
    loss_te = np.mean(Losses_te)
    var_tr = np.std(Losses_tr)
    var_te = np.std(Losses_te)

    return loss_tr,loss_te, var_tr,var_te

    
#---------------------------------RUN & PLOT-------------------------------------------

def cross_validation_visualization(lambds, mse_tr, mse_te, std_tr, std_te):
    """visualization the curves of mse_tr and mse_te."""
    
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.errorbar(lambds,mse_tr,std_tr,linestyle='None', color='b', marker='o')
    plt.errorbar(lambds,mse_te,std_te, linestyle='None', color='r',marker='o')
    plt.xlabel("lambda")
    plt.ylabel("errors made %")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)

def cross_val_run(y, tx, Features, Degrees, Lambdas, k_fold =5, seed =1)
    fig=plt.figure(3) #final plot % errors made vs. polynomial Degree for different combinations

    #iterate over different feature expansions combinations, polynomial degrees, and 
    #ridge regression hyperparameter lambda_
    for cross in Features:
        for log in Features:

            
            
            
            Min_loss_Degrees=[] #list of minimum loss over ridge parameter for
                                #different degrees
            
            Lambda_min=[]    #ridge parameter lambda corresponding to above minimum losses
            Min_var=[]       #same for variance

            for degree in Degrees:
                # define lists to store the loss and Variance of training data and test data
                #for a given degree and Features combination but over ridge parameter lambda_
                Loss_tr = []
                Loss_te = []
                Var_tr= []
                Var_te= []
                min_loss = 1e10  #temporary min loss
                lambda_min = 0   # temporary ____
                min_var = 0      #_____________

                for lambda_ in Lambdas: #ridge parameter scan
                    #calculate using cross validation and ridge regression
                    loss_tr, loss_te, var_tr, var_te= cross_validation(y, tx, 
                                                      k_fold,lambda_, degree,seed,
                                                      'ridge',cross,log)
                        
                    Loss_tr += [loss_tr]  #update losses list for every lamda_
                    Loss_te += [loss_te]  
                    Var_tr += [var_tr]
                    Var_te += [var_te]
                        
                    if loss_te < min_loss: #update best loss if improved
                        lamda_min = lambda_ #save corresponding lambda_
                        min_var=var_te      #save corresponding variance
                        min_loss = loss_te
                
                
                Min_loss_Degrees += [min_loss] #update list of minimum loss over 
                                               #ridge parameter lambda for
                                               #different degrees
                                               
                                                  
                Lambda_min += [lambda_min]   #corresponding lambda's and Variances
                Min_var += [min_var]         #at that min loss
                    
                    
                # Plot the obtained results % of errors made vs. lambda_
                plt.figure(1)
                cross_validation_visualization(Lambdas, Loss_tr, Loss_te, Var_tr, Var_te)
                title = 'cross validation, degree= ' +str(degree) 
                t2=''
                if log == 'true':
                    title += ', log terms'
                    t2 += 'log'
                if cross == 'true':
                    title += ', cross terms'
                    t2+= 'cross'
                    
                plt.title(title)
                plt.savefig('c_v_deg='+str(degree)+t2)
                plt.show()
                    


            # plot min losses(here average of %of errors made on test set) 
            # achieved for each degree and wait until the end to save multiple curves
            #for multiple different features combinations 
            plt.figure(3)
            
            labl='poly'   
            if log == 'true':
              
                labl += 'log_+'
                
            if cross == 'true':
                labl += '_cross'
            plt.errorbar(Degrees, Min_loss_Degrees,Min_var, marker='o', label=labl)
                
            #keep the global minimum loss over all degrees and ridge parameter
            #and corresponding degree, lambda and minimum variance
            global_min = min(Min_loss_Degrees) 
            global_min_index = Min_loss_Degrees.index(global_min)    
            
            min_degree = Degrees[global_min_index]
            global_lambda_min = Lambda_min[global_min_index]
            min_var= Min_var[global_min_index]
            
            #save all to txt file
            feature = 'polynomial'
            if log == 'true':
                feature += ' + log'
            if cross == 'true':
                feature += '+ cross'
                
            f = open("best_methods.txt","a") #append mode 

            f.write('for ' + feature + ':' + '\n')
            f.write('         min_degree = '+ str(min_degree) + 'min_loss = ' 
                   + str(global_min) + '  min_std = ' + str(min_var) 
                   + '  lambda_min = ' + str(global_lambda_min))
            f.write('\n \n')
                
            f.close()
                
    #final plot of best losses over degrees for every feature expansion combination      
    plt.title('best features, ridge regression')
    plt.xlabel("polynomial degree")
    plt.ylabel("loss")
    plt.legend(loc='best') 
    fig.savefig('best_features_for_ridge_regression')
        
          

    
      
  