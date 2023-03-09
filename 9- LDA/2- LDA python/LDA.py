import numpy as np
class LDA:
    def __init__(self,n_components):
        self.n_components=n_components
        self.LDA_vectors=None
    def fit(self,X,y):
        n_samples,n_features=X.shape
        X_mean=np.mean(X,axis=0)
        n_classes=np.unique(y)
        S_W=np.zeros((n_features,n_features))
        S_B=np.zeros((n_features,n_features))

        for c in n_classes:
            X_c= X[c==y]
            XC_mean=np.mean(X_c,axis=0)
            S_W+=np.dot((X_c-X_mean).T,(X_c-XC_mean))
            mean_diff=X_mean-XC_mean
            n_sub_samples=X_c.shape[0]
            S_B+=n_sub_samples* np.dot(mean_diff,mean_diff.T)
        Z=np.linalg.inv(S_W).dot(S_B)
        eigen_values,eigen_vectors=np.linalg.eig(Z)
        eigen_vectors=eigen_vectors.T
        idx=np.argsort(eigen_values)[::-1]
        eigen_values=eigen_values[idx]
        eigen_vectors=eigen_vectors[idx]
        self.LDA_vectors=eigen_vectors[0:self.n_components]
    def transform(self,X):
        return np.dot(X,self.LDA_vectors.T)