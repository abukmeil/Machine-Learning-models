import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from DecisionTree import DecisionTree
from collections import Counter
np.random.seed(1234)

# Load data
data=load_breast_cancer()
X,y=data.data, data.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=True, stratify=y,random_state=1234)
print(f' The shape of the X_train is: {X_train.shape}')
print(f' The shape of the X_test is: {X_test.shape}')
print('')
counter=Counter(y_train)
print(fr'The distribution of data samples fro class 0 and class 1 {counter}')


# Visualize the first two feature vectors
font1 = {'family':'serif','weight':'bold','color':'black','size':14}
cmap=ListedColormap(['#FF0000','#00FF00'])
fig,ax=plt.subplots(1,figsize=(6,4))
ax.scatter(X[:,0],X[:,1],c=y,cmap=cmap, s=15,)
plt.title('The first two feature vectors',fontdict=font1)
plt.xlabel('X_1',fontdict=font1)
plt.ylabel('X_2',fontdict=font1)
for tag in ['Malignant', 'Benign']:
    plt.scatter([], [], c='r' if tag == 'Malignant' else 'g', alpha=0.4, s=25,
                label=tag)
plt.legend(scatterpoints=1, frameon=True, labelspacing=0.5, title='Classes')
plt.show(block=False)
plt.pause(3)
plt.close()


# Classifier instance building
clf=DecisionTree(max_depth=20)
clf.fit(X_train,y_train)

# Prediction
y_predicted=clf.predict(X_test)
print(f'The model accuracy is : {clf.model_accuracy(y_test,y_predicted)}')
print('')
print(f'The model accuracy report is: {classification_report(y_test,y_predicted)}')
print('')
print(f'The model confusion matrix is: {confusion_matrix(y_test,y_predicted)} ')

