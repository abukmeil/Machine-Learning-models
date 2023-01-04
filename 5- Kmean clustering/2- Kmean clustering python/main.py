import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from Kmean_clustering import KMean
X,y=datasets.make_blobs(n_samples=300,n_features=2,centers=4, cluster_std=1.02,shuffle=True,random_state=1234)
print(X.shape)
print(y.shape)

#fig=plt.figure(figsize=(6,3))
#plt.scatter(X[:,0],X[:,1],c=y)
#plt.show(block=False)
#plt.pause(0.1)
#plt.close()


kmean=KMean()
y_predicted=kmean.predict(X)
print(y_predicted)
