import matplotlib.pyplot as plt

#f(w)=w**2+2w+c error function
def error(w):
	return (w**2)+(2*w)+2

def gradient(w):
	return (2*w)+2

w=range(-10,10)
err=[]
gradients=[]

for i in w:
	err.append(error(i))
	gradients.append(gradient(i))
	pass

plt.plot(w,err)
plt.plot(w,gradients,c='r')
plt.show()