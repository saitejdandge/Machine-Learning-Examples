import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stat
import time
data=pd.read_csv('Datasets/regr01.txt',sep=" ",header=None,names=['l','t'])

l=data['l'].values
t=data['t'].values

plt.xlabel("L")
plt.ylabel("t")
tsq=t**2
plt.plot(l,t**2)


m=0
c=0
lr=0.001


def train(x,y,m,c,eta):

	const=-(2.0/len(y))
	y_predicted=m*x+c
	delta_m=const*(sum(x*(y-y_predicted)))
	delta_c=const*(sum(y-y_predicted))
	m=m-delta_m*eta
	c=c-delta_c*eta
	err=sum((y-y_predicted))**2/len(y)
	return m,c,err

def train_on_all(x,y,m,c,eta,iterations=1000):
	for steps in range(iterations):
		m,c,err=train(x,y,m,c,eta)
	return m,c,err

	# Training for 1000 iterations, plotting after every 100 iterations:
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
plt.ion()
fig.show()
fig.canvas.draw()

for num in range(10):
    m, c, error = train_on_all(l, tsq, m, c, lr, iterations=100)
    print("m = {0:.6} c = {1:.6} Error = {2:.6}".format(m, c, error))
    y = m * l + c
    ax.clear()
    ax.plot(l, tsq, '.k')
    ax.plot(l, y)
    fig.canvas.draw()
    time.sleep(1)