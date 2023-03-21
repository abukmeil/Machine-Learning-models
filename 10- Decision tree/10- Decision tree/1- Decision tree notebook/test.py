import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from DecisionTree import DecisionTree

dataset=load_breast_cancer()
dataset.keys()
X,y=dataset.data , dataset.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,shuffle=True,random_state=1234)

clf=DecisionTree(max_depth=5)
clf.fit(X_train,y_train)
y_predicted=clf.predict(X_test)
print(y_predicted)