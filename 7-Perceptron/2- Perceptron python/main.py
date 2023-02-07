import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from Perceptron import Perceptron

cmap=ListedColormap(["#FF0000","#00FF00"])
X,y=datasets.make_blobs(n_samples=500,n_features=2,cluster_std=1.02,centers=2,shuffle=True,random_state=1234)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5,shuffle=True,random_state=1324)
fig,ax=plt.subplots(1,figsize=(5,3))
ax.scatter(X[:,0],y,c=y,cmap=cmap,marker='o',s=12)
plt.show(block=False)
plt.pause(3)
plt.close()

clf=Perceptron(lr=0.001,n_iteration=2000)
clf.fit(X_train,y_train)
y_pedicted=clf.predict(X_test)
print(clf.accuracy(y_test,y_pedicted))

fig,ax=plt.subplots(1,figsize=(5,3))
ax.scatter(X[:,0],X[:,1],c=y,cmap=cmap,marker='o')
lower_x_coordinate=np.amin(X[:,0])
higher_x_coordinate=np.amax(X[:,0])
x1_1_hyper_plane=clf.hyperplane_y_coordinate(lower_x_coordinate,clf.weights,clf.bias,0)
x1_2_hyper_plane=clf.hyperplane_y_coordinate(higher_x_coordinate,clf.weights,clf.bias,0)
ax.plot([lower_x_coordinate,higher_x_coordinate],[x1_1_hyper_plane,x1_2_hyper_plane])
plt.show(block=False)
plt.pause(3)
plt.close()


