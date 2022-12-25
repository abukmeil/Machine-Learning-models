import numpy as np
import sklearn
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from LogisticRegressor import LogisticRegressor
from Loss_accuracy_metrics import BCE_loss, Accuracy_metric
from sklearn.model_selection import train_test_split


'Importing the dataset'
Breast_dataset=datasets.load_breast_cancer()
X,y=Breast_dataset.data, Breast_dataset.target
print(f"The dataset shape is {X.shape}")
print(f"The targets shape is {y.shape}")

'Visualizing the trends of the data samples'
fig=plt.figure(figsize=(7,4))

font1 = {'family': 'serif', 'weight': 'bold', 'color': 'black', 'size': 14}
font2 = {'family': 'serif', 'weight': 'bold', 'color': 'black', 'size': 20}
plt.scatter(X[:,0],y,color='blue',marker='o')
plt.ylabel('y_true',fontdict=font1)
plt.xlabel('X[:,0]',fontdict=font1)
plt.title("The first feature of the data vs. labels",fontdict=font2)
plt.show(block=False)
plt.pause(2)
plt.close()

'Splitting the dataset'
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=True,random_state=1234)
print(f"The training set shape is {X_train.shape}")
print(f"The training target shape is {y_train.shape}")

'Building the classifier'
clf=LogisticRegressor(lr=0.0001,n_iteration=150)
clf.fit(X,y)
y_predicted=clf.predict(X)

'Measuring thg classification loss'
loss=BCE_loss(y,y_predicted)
print(f"The mean binary cross entropy is {loss}")

'Measuring the classification accuracy'
Accuracy=Accuracy_metric(y,y_predicted)
print(f"The classification accuracy is {Accuracy}")


