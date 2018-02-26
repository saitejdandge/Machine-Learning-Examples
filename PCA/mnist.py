from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot  as plt
df=load_digits()
scaler=StandardScaler()
newData=scaler.fit_transform(df.data)
accuracies=[]
components=[]

for i in xrange(1,newData.shape[1]):
	pca=PCA(n_components=i)
	pcaData=pca.fit_transform(newData)
	X_train,X_test,y_train,y_test=train_test_split(pcaData,df.target,random_state=0)
	clf=KNeighborsClassifier()
	clf.fit(X_train,y_train)
	components.append(i)
	accuracies.append(clf.score(X_test,y_test))

	pass

plt.ylabel("Accuracies")
plt.xlabel("PCA components")
print(str(components))
print(str(accuracies))
plt.plot(components,accuracies,'r+')






pca=PCA(n_components=2)
pca.fit(df.data)
plt.matshow(pca.components_,cmap='viridis')
plt.yticks([0,1],["PC1","PC2"])
plt.xticks(range(len(df.target)),range((newData.shape[1])),rotation=60,ha='left')
plt.xlabel("Feature")
plt.ylabel("Principle components")
plt.show()