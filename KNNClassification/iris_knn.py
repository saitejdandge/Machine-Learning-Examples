from sklearn import datasets

#importing iris dataset 
iris=datasets.load_iris()

#iris.data has features, iris.target has labels
print(iris.data)
print(iris.target)

#each entry has 4 attributes, 

#Spliting data into training and testing sets

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(iris.data,iris.target,test_size=0.33)

#0.33 means whole set has been divided in to two sets where test set is 33% of original set and train set in 66% of original test 

#lets import knn classifier 

from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors=3).fit(x_train,y_train)

#now we have trained our classifier using training data
#infact knn is a lazy learner, in reality it doesnt get trained , it takes all data with it when testing each attribute


#lets import accuracy metric

from sklearn.metrics import accuracy_score
print("accuracy is ")
print(accuracy_score(y_test,clf.predict(x_test)))

#we got accuracy of 96 when we use k=3, now lets try to plot graph with k and accuracy


import matplotlib.pyplot as plt

#we now iterate our classifier and init it different k values and find accuracy

#accuracy values is 2D array, where each entry is [K,accuracy]
accuracy_values=[]

for x in xrange(1,x_train.shape[0]):
	clf=KNeighborsClassifier(n_neighbors=x).fit(x_train,y_train)
	accuracy=accuracy_score(y_test,clf.predict(x_test))
	accuracy_values.append([x,accuracy])
	pass

#converting normal python array to numpy array for some efficient operations

import numpy as np
accuracy_values=np.array(accuracy_values)

plt.plot(accuracy_values[:,0],accuracy_values[:,1])
plt.xlabel("K")
plt.ylabel("accuracy")
plt.show()


# you see accuracy drops when k is more than 60
# K value selection depends upon data distribution and in this case, we have good accuracy when k lies between 40-60




#Thank you Happy Learning











