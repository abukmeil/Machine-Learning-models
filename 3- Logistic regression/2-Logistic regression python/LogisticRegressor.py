import numpy as np
import warnings



class LogisticRegressor:

    def __init__(self,lr=0.001,n_iteration=1000):
        self.ler=lr
        self.n_iteration=n_iteration
        self.weights=None
        self.bias=None
    def fit(self,X,y):
        n_sample,n_features=X.shape
        self.weights=np.zeros(n_features)
        self.bias=0

        for _ in range(self.n_iteration):
            linear_modeling=np.dot(X,self.weights)+self.bias
            y_predicted=self._sigmoid(linear_modeling)
            dw=(1/n_sample)*np.dot(X.T,(y_predicted-y))
            db=(1/n_sample)* np.sum(y_predicted-y)
            self.weights-=self.ler*dw
            self.bias-=self.ler*db

    def predict(self,X):
        linear_modeling = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_modeling)
        y_predicted_class=[1 if i >0.5 else 0 for  i in y_predicted]
        return np.array(y_predicted_class)

    def _sigmoid(self,x):
        # We use warning because exp show error "RuntimeWarning: overflow encountered in exp"
        warnings.filterwarnings('ignore')

        return 1/(1+np.exp(-x))
