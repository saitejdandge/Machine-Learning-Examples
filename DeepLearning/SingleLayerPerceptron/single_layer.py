# Just disables the warning, doesn't enable AVX/FMA
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

input_size=0

output_size=0




def get_sparse_matrix(list_value):

	list_value=list_value.tolist()
	set_value=set(list_value)
	
	outer=[]
	k=0
	total=len(list_value)*len(set_value)
	for i in list_value:
		
		inner=[]
		print("----------------Parsing Output-----------------")
		percent=((k/total)*100)
		print(percent)
		
		for j in set_value:

			if i == j:
				inner.append(1)
			else :
				inner.append(0)
				pass

			k+=1
			pass
		outer.append(inner)
		
		pass

	return np.array(outer)

	pass


def preprocess(data_frame):
	
	input_size=data_frame.data.shape[1]
	
	output_matrix=get_sparse_matrix(data_frame.target)
	
	output_size=output_matrix.shape[1]

	return input_size,output_size,output_matrix





from sklearn.datasets import load_iris

data_frame=load_iris()

input_size,output_size,output_matrix= preprocess(data_frame)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(data_frame.data,output_matrix)



def get_batch(X_train,batch_size):

	features_batch=[]

	labels_batch=[]

	random_index=np.random.randint(0,X_train.shape[0]-1)


	for i in range(0,batch_size):

		features_batch.append(X_train[random_index])
		
		labels_batch.append(y_train[random_index])
		pass


	return features_batch,labels_batch
	
	pass






import matplotlib.pyplot as plt

import tensorflow as tf


sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, input_size])
y_ = tf.placeholder(tf.float32, shape=[None, output_size])

W = tf.Variable(tf.zeros([input_size,output_size]))
b = tf.Variable(tf.zeros([output_size]))


sess.run(tf.global_variables_initializer())

y = tf.matmul(x,W) + b

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


testing_accuracies=[]


iterations=range(1,5000)

for i in iterations:
  #batch = mnist.train.next_batch(100)
  print("Traning Iteration ")
  print(i)
  print("-------------------------------------------")
  features_batch,labels_batch=get_batch(X_train,100)
  train_step.run(feed_dict={x: features_batch, y_: labels_batch})

  
  # correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
  # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  # test_score=(accuracy.eval(feed_dict={x: X_test.tolist(), y_: y_test.tolist()}))
  # print(test_score)
  # testing_accuracies.append(test_score)


  
  # plt.scatter(i,test_score,color='blue')
  # plt.plot(iterations[:i],testing_accuracies,C='blue')
  # plt.xlabel("Iterations")
  # plt.ylabel("Accuracies")
  # plt.pause(0.0000005)

  pass


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_score=(accuracy.eval(feed_dict={x: X_test.tolist(), y_: y_test.tolist()}))
print(test_score)
print("-------done-----------")



