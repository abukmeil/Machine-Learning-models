import numpy as np


def BCE_loss(y_true,y_predicted):
    y_class0_loss=y_true*np.log(y_predicted+1e-9)
    y_class1_loss=(1-y_true)*np.log(1-y_predicted+1e-9)
    return -np.mean(y_class0_loss+y_class1_loss)

def Accuracy_metric(y_true,y_predicted):
    return np.sum(y_true==y_predicted)/len(y_true)