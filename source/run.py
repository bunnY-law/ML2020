from proj1_helpers import *
from implementations import *

path = '../data/train.csv'
y, tx,ids = load_csv_data(path)

DATA_TEST_PATH = '../data/test.csv' #download train data and supply path here 
y_test, tx_test, ids_test = load_csv_data(DATA_TEST_PATH)
print('loaded y_test, tx_test from ' + DATA_TEST_PATH)



degree = 2
method = 'mean' #apply mean when standardizing data (pre_process)
cross='true' #add cross multiplications terms
log='true'



#pre_process train data 
tx = pre_process(tx,method,degree,cross,log)

# Have to apply the same data cleaning as for train set
tx_test = pre_process(tx_test,method,degree,cross,log)


lambda_=1e-8
#train model
w_trained,_=ridge_regression(y,tx,lambda_)
#test accuracy of predictions on itself (just to see if training went ok)
print('wrong: ', err_percent(y,tx,w_trained) , '%')


#predict y labels +1 or -1 with trained w
y_pred = predict_labels(w_trained, tx_test) 

OUTPUT_PATH = '../data/result.csv' # fill in desired name of output file for submission

create_csv_submission(ids_test, y_pred, OUTPUT_PATH) #save preds to output path


print('created result.csv in ' + OUTPUT_PATH)