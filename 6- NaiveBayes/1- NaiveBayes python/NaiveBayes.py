import numpy as np
class NaiveBayes:
    # No constructor, __init__ is required

    def fit(self,X,y):
        n_samples,n_features=X.shape
        self._classes=np.unique(y)
        n_classes=len(self._classes)

        # Initiating prior, mean, variance
        self._prior=np.zeros(n_classes,dtype=np.float64)
        self._mean=np.zeros((n_classes,n_features),dtype=np.float64)
        self._variance=np.zeros((n_classes,n_features),dtype=np.float64)

        # Iterate over data to select samples for each class using mask filtering
        for class_ in self._classes:
            data_class=X[class_==y]
            self._prior[class_]= data_class.shape[0]/float(n_samples)
            self._mean[class_,:]=data_class.mean(axis=0)
            self._variance[class_,:]=data_class.var(axis=0)

    def predict(self,X):
        y_predicted=[self._predict(x) for x in X]
        return y_predicted

    def _predict(self,x):
        posteriors=[]
        for idx, class_ in enumerate(self._classes):
            prior=np.log(self._prior[idx])
            class_conditional_prob= np.sum(np.log(self._Gaussian_dist(x,idx)))
            posterior=prior+class_conditional_prob
            posteriors.append(posterior)
        return self._classes[np.argmax(posteriors)]

    def _Gaussian_dist(self,x,idx):
        mean=self._mean[idx]
        variance=self._variance[idx]
        nominator=np.exp(-(x-mean)**2 /2*variance)
        denominator=np.sqrt(2*np.pi*variance)
        pdf=nominator/denominator
        return pdf




