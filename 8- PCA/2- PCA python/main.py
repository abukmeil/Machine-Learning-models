import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from PCA import PCA

iris=datasets.load_iris()
print(iris.keys())
X=iris.data
y=iris.target  # Only for visualization not for training
print(X.shape)

# Visualizing the first two feature vectors
cmap=ListedColormap(['#FF0000','#00FF00','#0000FF'])
fig,ax=plt.subplots(1,figsize=(5,3))
ax.scatter(X[:,0],X[:,1],c=y,cmap=cmap,s=15,marker='o')
ax.axes.get_xaxis().set_ticks([])
ax.axes.get_yaxis().set_ticks([])
plt.title("Scatter plot of the first two features")
plt.show(block=False)
plt.pause(2)
plt.close()

# PCA transform
PCA_transform=PCA(n_components=2)
PCA_transform.fit(X)
projection=PCA_transform.transform(X)
print(projection.shape)

# Visualizing the projected data

font1 = {'family':'serif','weight':'bold','color':'black','size':12}
cmap=ListedColormap(['#FF0000','#00FF00','#0000FF'])
fig,ax=plt.subplots(1,figsize=(6,4))
ax.scatter(projection[:,0],projection[:,1],c=y,s=15,marker='o',cmap=cmap)
ax.axes.get_xaxis().set_ticks([])
ax.axes.get_yaxis().set_ticks([])
plt.title("Scatter plot of the first two PCAs",fontdict=font1)
plt.xlabel('PCA#1',fontdict=font1)
plt.ylabel('PCA#2',fontdict=font1)
plt.show()
