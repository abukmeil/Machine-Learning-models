import numpy as np
from Euclidean_distance import Euclidean_distance
from collections import Counter

class KNNclassifier:
    def __init__(self,k=3):
        # stor k value
        self.k=k

    # knn implies no training, just storing x_train, y_train
    def fit(self,X,y):
        self.X_train=X
        self.y_train=y

    # X because it contains multiple x
    def predict(self,X):
        # use helper _predict because of multiple sample prediction
        predicted_labels=[self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self,x):
        # Compute the Euclidean distances
        Eu_distance=[Euclidean_distance(x,x_train) for x_train in self.X_train]

        # Compute the K nearest samples
        K_nearst_indecies=np.argsort(Eu_distance)[:self.k]

        K_nearst_lables=[self.y_train[i] for i in K_nearst_indecies]

        # Majority vote of the nearest samples to have the most common class label

        major_class=Counter(K_nearst_lables).most_common(1)
        return major_class[0][0]


