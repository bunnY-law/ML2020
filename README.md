# Machine Learning Project 1 - Higgs Boson

## Team Members
* Dario Müller
* Lawrence Brun
* Florian Genilloud

## Introduction 

The goal of this project is to implement different regression method using numpy. In addition, as it is the first Machine Learning project,
it helps us experiment basic features engineering and cross-validation to find the best hyperparameters for each method.

## Stucture of the folder

* `source/implementation.py`: Which contains all the regression Method : Least Squares regression (Standard, GD, SGD), Ridge Regression and Logistic Regression (Normal and Regularized) in addition, it also contains all the cost and gradient for each method as well as the cross-validation function and the grid search.
* `source/proj1.ipynb`: Notebook with the exploratory of the dataset and all the computations.
* `source/proj1_helpers.py`: Contains the functions provided for the project
* `data/test.csv`: Contains the test set
* `data/train.csv`: Contains the train set
* `data/result.csv` : Contains the best result obtained during our project

## How to run:
The modules required are : `numpy`, `matplotlib.pyplot`.
The best method parameters are saved in : `best_methods.txt`
To see the evolution's plots of the accuracy for the best method, please look at the file : `c_v_deg=2logcross`

1. If you want to reproduce the best result obtained and submitted on [AIcrowd](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/leaderboards):
```
python source/run.py
```

