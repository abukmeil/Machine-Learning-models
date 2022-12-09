############## This is main file to run KNN ################

'Importing the main libraries'
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from KNN import KNNclassifier
from Accuricy_comparsion import mean_acc_metric

'RGB color mapping'
cmap=ListedColormap(['#FF0000','#00FF00','#0000FF'])

if __name__=='__main__':

    'Load iris dataset'
    iris=datasets.load_iris()
    print(iris.keys())
    X,y=iris['data'],iris['target']
    'Check the shape of the data'
    print(X.shape)
    print(y.shape)


    'Visualizing the first two features of the data using scatter plot'
    fig=plt.figure(figsize=(7,4))
    plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap,s=20)
    plt.show()

    'Spliting dataset in train and test datasets'
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=1234)

    'Identify a classifier '

    clf=KNNclassifier(k=3)
    clf.fit(X_train,y_train)
    KNN_prediction=clf.predict(X_test)
    acc=mean_acc_metric(y_test,KNN_prediction)
    print(acc)
