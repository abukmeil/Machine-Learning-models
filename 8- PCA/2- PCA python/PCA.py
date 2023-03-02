import numpy as np

class PCA:
    def __init__(self,n_components):
        self.n_components=n_components
        self.mean=None
        self.PCA_vectors=None
    def fit(self,X):
        self.mean=np.mean(X,axis=0)

        #Covaraince
        cov_x=np.cov(X.T)

        # Eigen-decomposition
        eign_value,eign_vector=np.linalg.eig(cov_x)
        eign_vector=eign_vector.T
        idx=np.argsort(eign_value)[::-1]
        eign_value=eign_value[idx]
        eign_vector=eign_vector[idx]
        self.PCA_vectors=eign_vector[:self.n_components]

    def transform(self,X):
        X=X-self.mean
        print(f' The shape of the data is: {X.shape}')
        print(f'The shape of the eigenvectors matrix to project into: {self.PCA_vectors.shape}')
        # Projection
        PCA_transform=np.dot(X,self.PCA_vectors.T)
        return PCA_transform