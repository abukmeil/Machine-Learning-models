import numpy as np
import matplotlib.pyplot as plt

class SVM:
    def __init__(self,lr=0.008,n_iteration=3000,lambda_reg=0.01):
        self.lr=lr
        self.n_iteration=n_iteration
        self.lambda_reg=lambda_reg
        self.weights=None
        self.bias=None

    def fit(self,X,y):
        self.X=X
        self.y=y
        y_=np.where(y<=0,-1,1)
        n_samples,n_features=X.shape
        self.weights=np.zeros(n_features)
        self.bias=0

        self.x0_1 = np.amin(X[:, 0])
        self.x0_2 = np.amax(X[:, 0])

        fig=plt.figure(figsize=(7,4))
        self.ax=fig.add_subplot()
        font1 = {'family': 'serif', 'weight': 'bold', 'color': 'black', 'size': 14}
        font2 = {'family': 'serif', 'weight': 'bold', 'color': 'black', 'size': 20}
        plt.scatter(X[:, 0], X[:, 1], c=y)

        for _ in range(self.n_iteration):
            for idx,x_i in enumerate(X):
                sign_check=y_[idx]*(np.dot(x_i,self.weights)-self.bias)>=1
                if sign_check:
                    self.weights-=self.lr*(2*self.lambda_reg*self.weights)
                    dw1=2 * self.lambda_reg * self.weights
                else:
                    self.weights-=self.lr*(2*self.lambda_reg*self.weights-np.dot(x_i,y_[idx]))
                    dw1=2 * self.lambda_reg * self.weights - np.dot(x_i, y_[idx])
                    self.bias-=self.lr*y_[idx]

            plt.scatter(X[:, 0], X[:, 1], c=y)

            self.x1_1 = self._get_hyperplane_value(self.x0_1, self.weights, self.bias, 0)
            self.x1_2 = self._get_hyperplane_value(self.x0_2, self.weights, self.bias, 0)
            self.x1_n = self._get_hyperplane_value(self.x0_1, self.weights, self.bias, -1)
            self.x2_n = self._get_hyperplane_value(self.x0_2, self.weights, self.bias, -1)
            self.x1_p = self._get_hyperplane_value(self.x0_1, self.weights, self.bias, 1)
            self.x2_p = self._get_hyperplane_value(self.x0_2, self.weights, self.bias, 1)

            self.ax.plot([self.x0_1, self.x0_2], [self.x1_1, self.x1_2],'--r')
            self.ax.plot([self.x0_1, self.x0_2], [self.x1_n, self.x2_n],'k')
            self.ax.plot([self.x0_1, self.x0_2], [self.x1_p, self.x2_p],'k')
            x1_min = np.amin(X[:, 1])
            x1_max = np.amax(X[:, 1])
            self.ax.axes.get_xaxis().set_ticks([])
            self.ax.axes.get_yaxis().set_ticks([])
            plt.xlabel("x1", fontdict=font1)
            plt.ylabel("x2", fontdict=font1)
            plt.title("SVM by: Dr. Mohanad Abukmeil", fontdict=font2)
            plt.pause(0.01)
            plt.cla()
        plt.show()


    def _get_hyperplane_value(self,x, w, b, shift):
        return (-w[0] * x + b + shift) / w[1]

    def predict(self,X):
        predicted=np.dot(X,self.weights)-self.bias
        return np.sign(predicted)