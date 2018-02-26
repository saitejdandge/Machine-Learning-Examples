from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer=load_breast_cancer()
import numpy as np
x_train,x_test,y_train,y_test=train_test_split(cancer.data,cancer.target,random_state=1)
print(x_train)
print(x_test.shape)


import matplotlib.pyplot as plt
plt.plot(x_train,np.ones(x_train.shape[0]),'r+')



from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(x_train)
x_scaled=scaler.transform(x_train)
plt.plot(x_scaled,np.ones(x_train.shape[0])+2,'y+')

plt.show()