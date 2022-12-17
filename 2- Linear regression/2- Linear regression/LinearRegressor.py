import numpy as np
class LinearRegressor:
    def __init__(self,lr=0.001,n_iteration=1000):
        self.lr=lr
        self.n_iteration=n_iteration
        self.weights=None
        self.bias=None
    def fit(self,X,y):
        n_samples,n_features=X.shape
        self.weights=np.zeros(n_features)
        self.bias=0

        for _ in range(self.n_iteration):
            y_predicted=np.dot(X,self.weights)+self.bias
            dw=(1/n_samples)*np.dot(X.T,(y_predicted-y))
            db=(1/n_samples)*np.sum(y_predicted-y)
            self.weights-=self.lr*dw
            self.bias-=self.bias*db

    def predict(self,X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted

