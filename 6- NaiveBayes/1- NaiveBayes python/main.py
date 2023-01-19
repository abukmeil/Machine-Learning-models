import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from NaiveBayes import NaiveBayes
from accuracy_metric import accuracy_metric

X,y=datasets.make_classification(n_samples=1000,n_features=10,n_classes=2,random_state=1234)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)


fig,ax=plt.subplots(1,1)
ax.scatter(X[:,0],X[:,1],c=y,marker='o',s=20)
ax.axes
ax.axes.get_xaxis().set_ticks([])
ax.axes.get_yaxis().set_ticks([])
ax.set_xlabel('Feature1', size=14,weight='bold')
ax.set_ylabel('Feature2', size=14, weight='bold')

plt.show(block=False)
plt.pause(1.5)
plt.close()

clf=NaiveBayes()
clf.fit(X_train,y_train)
y_predicted=clf.predict(X_test)

accuracy=accuracy_metric(y_test,y_predicted)
print(accuracy)

