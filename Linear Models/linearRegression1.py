import numpy as np
import pandas as pd
import scipy.stats as stat
import matplotlib.pyplot as plt


data=pd.read_csv("Datasets/regr01.txt", sep=" ", header=None, names=["x","y"])
print(data.head())
print(data.tail())

x=data['x'].values
y=data['y'].values


y=y**2

plt.plot(x,y,'r+')

m,c,_,_,_=stat.linregress(x,y)
ylr = m * x + c
plt.plot(x,ylr)



legend=["points(x,y**2)","line"]
for i in xrange(1,10):
	
	y[3]=y[3]+i
	m,c,_,_,_=stat.linregress(x,y)
	ylr = m * x + c
	legend.append('error '+str(i))
	plt.plot(x,ylr)


plt.legend(legend)
plt.show()