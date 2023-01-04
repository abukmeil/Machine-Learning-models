'Importing requirements '
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from SVM import SVM
from accuracy_metrics import accuracy

'Creating dataset'
X,y=datasets.make_blobs(n_samples=200,n_features=2,centers=2,cluster_std=1.05,random_state=1234)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=True,random_state=1234)
'''
'Visualizing the trends among the data samples'
fig,ax=plt.subplots(1,2) #  one row and  two column
ax[0].scatter(X[:,0],X[:,1],c=y, marker='o',s=20)
plt.show(block=False)
plt.pause(0.1)
plt.close
'''
'Converting class 0 label to be -1'
y=np.where(y<=0,-1,1)

'Building the classifier'
clf=SVM()
clf.fit(X_train,y_train)
y_predicted=clf.predict(X_test)

'Printing the result'
print(clf.weights, clf.bias)
print(y_predicted)


def hyperplane_y_coordinate(x,w,b,shift):
    return (-w[0] * x + b + shift)/w[1]


lower_x_coordinate=np.amin(X[:,0])
higher_x_coordinate=np.amax(X[:,0])
x1_1_hyper_plane=hyperplane_y_coordinate(lower_x_coordinate,clf.weights,clf.bias,0)
x1_2_hyper_plane=hyperplane_y_coordinate(higher_x_coordinate,clf.weights,clf.bias,0)

x1_n=hyperplane_y_coordinate(lower_x_coordinate,clf.weights,clf.bias,-1)
x2_n=hyperplane_y_coordinate(higher_x_coordinate,clf.weights,clf.bias,-1)

x1_p=hyperplane_y_coordinate(lower_x_coordinate,clf.weights,clf.bias,1)
x2_p=hyperplane_y_coordinate(higher_x_coordinate,clf.weights,clf.bias,1)

'Plotting the fitting line'
fig,ax=plt.subplots()
font1 = {'family':'serif','weight':'bold','color':'black','size':14}
font2 = {'family':'serif','weight':'bold','color':'dimgray','size':20}

ax.scatter(X[:,0],X[:,1],c=y,marker='o')

ax.plot([lower_x_coordinate,higher_x_coordinate],[x1_1_hyper_plane,x1_2_hyper_plane],'--r')
ax.plot([lower_x_coordinate,higher_x_coordinate],[x1_n,x2_n],'k')
ax.plot([lower_x_coordinate,higher_x_coordinate],[x1_p,x2_p],'k')
ax.axes.get_xaxis().set_ticks([])
ax.axes.get_yaxis().set_ticks([])

plt.xlabel("x1",fontdict=font1)
plt.ylabel("x2",fontdict=font1)
plt.title("SVM by: Dr. Mohanad Abukmeil",fontdict=font2)
plt.show(block=False)
plt.pause(1)
plt.close()

accuracy_=accuracy(y_test,y_predicted)
print(f"The accuracy {accuracy_}")
