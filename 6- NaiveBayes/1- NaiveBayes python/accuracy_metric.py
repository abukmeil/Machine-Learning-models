import numpy as np
def accuracy_metric(y_true,y_predicted):
    return np.sum(y_true==y_predicted)/len(y_true)