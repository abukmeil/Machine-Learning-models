import numpy as np
def mean_acc_metric(y_true,y_predicted):
    return np.sum(y_true==y_predicted)/len(y_true)
