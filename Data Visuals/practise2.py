from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
load_irisDF=load_iris()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

X_train,X_test, y_train,y_test=train_test_split(load_irisDF.data,load_irisDF.target,random_state=0)


print(X_train.shape)
print(X_test.shape)
from sklearn.decomposition import PCA

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
pca=PCA(n_components=4)
X_pca=pca.fit_transform(X_train_scaled)

print(X_pca)

plt.plot(X_pca[:,0],X_pca[:,1],'r+')


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_pca,y_train)
print("Accuracy is ")

print(knn.score(pca.fit_transform(scaler.fit_transform(X_test)),y_test))
