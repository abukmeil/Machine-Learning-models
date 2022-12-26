
import numpy as np

class SVM:
    def __init__(self,lr=0.001,n_iteration=5000,lambda_reg=0.001):
        self.lr=lr
        self.n_iteration=n_iteration
        self.lambda_reg=lambda_reg
        self.weights=None
        self.bias=None

    def fit(self,X,y):
        y_=np.where(y<=0,-1,1)
        n_samples,n_features=X.shape
        self.weights=np.zeros(n_features)
        self.bias=0

        for _ in range(self.n_iteration):
            for idx,x_i in enumerate(X):
                sign_check=y_[idx]*(np.dot(x_i,self.weights)-self.bias) >=1
                if sign_check:
                    dw=2*self.lambda_reg*self.weights
                    self.weights-=self.lr*dw
                else:
                    dw=2*self.lambda_reg*self.weights-np.dot(x_i,y_[idx])
                    db=y_[idx]
                    self.weights-=self.lr*dw
                    self.bias-=self.lr*db

    def predict(self,X):
        y_predicted=np.dot(X,self.weights)-self.bias
        return np.sign(y_predicted)

