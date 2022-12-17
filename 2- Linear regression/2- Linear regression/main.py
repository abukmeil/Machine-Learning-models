# Importing required libraries
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from LinearRegressor import LinearRegressor
from mse_metric import mse

# Building dataset
X,y=datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=1234,shuffle=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)

# Visualizing the built dataset
cmap=plt.get_cmap('viridis')
fig=plt.figure(figsize=(7,4))
plt.scatter(X_train[:,0],y_train,color=cmap(0.5),marker='o',s=20)
plt.scatter(X_test[:,0],y_test,color=cmap(0.9),marker='o',s=20)
plt.show()

# Using the programmed regressor
regr=LinearRegressor(lr=0.01,n_iteration=10000)
regr.fit(X_train,y_train)
y_predicted=regr.predict(X_test)

# Computing Regression loss
loss=mse(y_test,y_predicted)
print(loss)

# Visualize the prediction
d_predict=regr.predict(X)
print(d_predict.shape)
cmap=plt.get_cmap('viridis')
fig=plt.figure(figsize=(7,4))
plt.scatter(X_train[:,0],y_train,color=cmap(0.5),marker='o',s=20)
plt.scatter(X_test[:,0],y_test,color=cmap(0.9),marker='o',s=20)
plt.plot(X,d_predict,color='k')
plt.show()