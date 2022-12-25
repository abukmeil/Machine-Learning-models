import numpy as np

def mse(y_true,y_predicted):
    return np.mean((y_true-y_predicted)**2)