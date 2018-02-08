"""
Hi all this is saitej
In this we are gonna classify handwritten digits using K nearest neighbor classifier
Lets go
"""

#import dataset in to program

from sklearn import datasets
digits=datasets.load_digits()



#we have 1797 records

#each entry is 8*8 matrix, we shall reshape this matrix to 1*64 so as to make our computation more human readable

digits.images=digits.images.reshape(digits.images.shape[0],digits.images.shape[1]*digits.images.shape[2])

#now lets print new matrix dimensions
print(digits.images.shape)



#lets try to print features and labels we have in the dataset
print(digits.images)
print(digits.target)

#lets also see the dimension of data we have

print(digits.images.shape)
print(digits.target.shape)



#now we have data, we need to split this whole chunk of data in to training and testing data set


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(digits.images,digits.target,test_size=0.25)

#We have split whole data in to 75% of training data and 25% of testing data

#lets import classifier and train it 

from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors=3).fit(x_train,y_train)

#we have given our classifier training set, using fit function.
#point to be remembered
#KNN is lazy learner, it takes whole data while classifying each test entry
#In other words, it doesnt learn anything at all. it justs returns nearest neighbors and give us mode of all nearest neighbors

#lets try to predict accuracy over test data

from sklearn.metrics import accuracy_score
print("accuracy found is")
print(accuracy_score(y_test,clf.predict(x_test)))
#our classifier is 98% accurate





















