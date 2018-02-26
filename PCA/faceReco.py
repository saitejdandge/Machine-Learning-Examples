from sklearn.datasets import fetch_lfw_people
import numpy as np
people=fetch_lfw_people(min_faces_per_person=20,resize=0.7,download_if_missing=True)
print(people.images.shape)


#Importing required packages and utilities

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt




people.target=people.target.reshape(people.target.shape[0],1)

#preprocessing data 
raw_data=people.images.reshape(people.images.shape[0],people.images.shape[1]*people.images.shape[2])
scaler=StandardScaler()
scaled_data=scaler.fit_transform(raw_data)

#spliting data into training and testing set



from sklearn.neighbors import KNeighborsClassifier




components=[]
accuracies=[]
for i in xrange(1,scaled_data.shape[1]):
	pca=PCA(n_components=i)
	pcaData=pca.fit_transform(scaled_data)
	clf=KNeighborsClassifier()
	X_train,X_test,y_train,y_test=train_test_split(pcaData,people.target,random_state=0)
	clf.fit(X_train,y_train)
	components.append(i)
	accuracy=clf.score(X_test,y_test)
	accuracies.append(accuracy)
	#print(i)
	#print(accuracy)
	#print("--------------")
	pass


plt.plot(components,accuracies,'r+')
plt.show()