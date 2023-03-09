import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from LDA import LDA
np.random.seed(1234)

iris=datasets.load_iris()
X=iris.data
y=iris.target

print(f'The data shape is {X.shape}')

# Visualizing the first two feature vectors
font1={'family':'serif','weight':'bold','color':'black','size':14}
cmap=ListedColormap(['#FF0000','#00FF00','#0000FF'])
fig,ax=plt.subplots(1,figsize=(6,4))
ax.scatter(X[:,0],X[:,1],c=y,cmap=cmap,s=15,marker='o',edgecolor='none')
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
plt.title("Scatter plot of the first two features",fontdict=font1)
plt.xlabel('x_1',fontdict=font1)
plt.ylabel('x_2',fontdict=font1)
plt.show(block=False)
plt.pause(3)
plt.close()

# LDA transform
lda=LDA(n_components=2)
lda.fit(X,y)
projected_data=lda.transform(X)
print(f'The original data before the projection is {X.shape}')
print(f'The projected data shape is {projected_data.shape}')

# Visualizing the projected data
fig,ax=plt.subplots(1,figsize=(6,4))
plt.scatter(projected_data[:,0],projected_data[:,1],c=y,cmap=cmap,s=15,marker='o',edgecolor='none')
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
plt.colorbar()
plt.title("Plot of the first two LDA components",fontdict=font1)
plt.xlabel("LDA component #1",fontdict=font1)
plt.ylabel("LDA component #1",fontdict=font1)
plt.show(block=False)
plt.pause(4)
plt.close()

