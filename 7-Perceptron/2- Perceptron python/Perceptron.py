import numpy as np
class Perceptron:
    def __init__(self,lr=0.001,n_iteration=500):
        self.lr=lr
        self.n_iteration=n_iteration
        self.weights=None
        self.bias=None

    def fit(self,X,y):
        y_=np.where(y>0,1,0)
        n_sample,n_feature=X.shape
        self.weights=np.zeros(n_feature)
        self.bias=0

        for _ in range(self.n_iteration):
            for idx,x_i in enumerate(X):
                linear_model=np.dot(x_i,self.weights)+self.bias
                y_predicted=self._activation_unit_step(linear_model)
                update_rule = (y_[idx] - y_predicted) * self.lr
                self.weights += update_rule * x_i
                self.bias += update_rule

    def predict(self,X):
        linear_model=np.dot(X,self.weights)+self.bias
        y_predicted=self._activation_unit_step(linear_model)
        return y_predicted

    def _activation_unit_step(self,lin_mod):
        return np.where(lin_mod>=0,1,0)

    @staticmethod
    def accuracy(y_true,y_predicted):
        return np.sum(y_true==y_predicted)/len(y_true)

    @staticmethod
    def hyperplane_y_coordinate(x,w,b,shift):
        return (-w[0]*x+b+shift)/w[1]




