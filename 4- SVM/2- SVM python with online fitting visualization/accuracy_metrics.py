import numpy as np

def accuracy(y_true,y_predicted):
    return np.sum(y_true==y_predicted)/len(y_true)