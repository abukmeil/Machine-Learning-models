import numpy as np
class LinRegression:
    
    
    def __init__(self,lr=0.01,num_itr=1000):
        self.lr=lr
        self.num_itr=num_itr
        self.weights=None
        self.bias=None
        self.gradient_weight=[]
        self.opt_weight=[]
        self.opt_prediction=[]


    
    def fit(self,X,y):
        # init parameters
        n_samples, n_features=X.shape
        self.weights=np.zeros(n_features)
        self.bias=0
        for _ in range(self.num_itr):
            y_predicted=np.dot(X,self.weights)+self.bias
            dw=(1/n_samples) * np.dot(X.T,(y_predicted-y))
            db=(1/n_samples) * np.sum(y_predicted-y)
            
            self.opt_weight.append(self.weights/1)
            self.weights-=self.lr*dw
            self.bias-=self.lr*db
            self.gradient_weight.append(dw)
            self.opt_prediction.append(y_predicted)
            
            
    def predict(self,X):
        y_predicted=np.dot(X,self.weights)+self.bias
        return y_predicted

    