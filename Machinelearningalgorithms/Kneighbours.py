import numpy as np
from sklearn import preprocessing,neighbors
from sklearn.model_selection import train_test_split
import pandas as pd



df= pd.read_csv('breastcancer')

df.replace('?', -99999, inplace=True)

#we dont want an id column as it does not have any significance related to
#benign and malignant.

#k nearest neighbor is worst for outlier

df.drop(['id'],1, inplace=True) #if we will include this id column then the accuracy
#will be almost 50% that is too worse for predicting a cancer


X = np.array(df.drop(['class'],1)) # df is everything but not class
y = np.array(df['class']) # y is just the class column

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)

#defining the classifier

clf=neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)


accuracy = clf.score(X_test, y_test)
print(accuracy)

#Now we are going to make prediction


example_measures = np.array([[4,2,1,1,1,2,3,2,1], [8,9,4,5,6,6,6,6,6]])
example_measures = example_measures.reshape(len(example_measures),-1)

#except id and class put some
#random numbers, make sure it is not occur in the train dataset

prediction = clf.predict(example_measures)
print(prediction)








