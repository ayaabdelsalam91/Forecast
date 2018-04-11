
from pandas import DataFrame
from pandas import read_csv
import random
import math
import numpy as np 
import tensorflow  as tf
from numpy import zeros, newaxis
from sklearn.model_selection import train_test_split
from scipy.ndimage.interpolation import shift
from Bias import getCenter
np.set_printoptions(suppress=True)
from helper import getNextMonth


Loc = '../Data/'
Model_Loc = '../Models/'




def unison_shuffled_copies_Correct(a, b,c):
    assert len(a) == len(b)
    assert len(a) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p] , c[p]


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def loss_function(y, outputs,seq_length , alpha):
    lasty = last_relevant(y, seq_length)
    lastOutput = last_relevant(outputs, seq_length)
    Lastloss = tf.reduce_sum(tf.abs(lasty -lastOutput))
    AllExceptloss =  tf.reduce_sum(tf.abs(y -outputs)) - Lastloss
    return tf.reduce_mean(AllExceptloss*alpha +Lastloss*(1-alpha))



def last_relevant(output, length):
    batch_size = tf.shape(output)[0]

    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    print("here" , batch_size , max_length , out_size)
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    print("relevant" , relevant)
    return relevant




def get_interval(MAE):
	return np.percentile(MAE,25),np.percentile(MAE,75)


def printstuff(x,y):
	for i in range(x.shape[0]):
		print ("Actual: " , x[i], " Precicted:  " , y[i])





def LSTM(OutputFile  , X_, Y_,Trainseq_length, TestData , Testseq_length ,
	learning_rate ,n_neurons, n_layers , alpha,n_epochs,  trainKeepProb = 1.0 , RID=None  , DS = 1 ,  testing=True,Bias=None ,hasMonth=True):
	print("Baseline")

	############################## LSTM Architecture #######################################
	

	n_steps = Y_.shape[1]
	n_inputs = X_.shape[2]
	n_outputs = Y_.shape[2]

	#create a placeholder for input and output
	X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
	y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
	seq_length = tf.placeholder(tf.int32, [None])
	keep_prob = tf.placeholder(tf.float32)
	

	#create lstm cells
	lstm_cells = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)
	              for layer in range(n_layers)]



	#make this LSTM recurrent 
	#At each time step we now have an output vector of size n_neurons.
	#But what we actually want is a single output value at each time step.
	#The simplest solution is to wrap the cell in an OutputProjectionWrapper.
	#A cell wrapper acts like a normal cell, proxying every method call to an underlying cell, but it also adds some functionality.
	#The OutputProjectionWrapper adds a fully connected layer of linear neurons (i.e., without any activation function)
	# on top of each output (but it does not affect the cell state).
	multi_cell = tf.contrib.rnn.OutputProjectionWrapper(
		tf.contrib.rnn.MultiRNNCell(lstm_cells),output_size=n_outputs)


	outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32, sequence_length=seq_length)
	# added by Zhe
	print(outputs[:,:,0].get_shape())
	
	rnn_outputs = tf.nn.dropout(outputs, keep_prob)
	loss = loss_function(y, rnn_outputs,seq_length,alpha)
	y_Last = last_relevant(y, seq_length)
	outputs_Last = last_relevant(rnn_outputs, seq_length)
	accuracy = tf.reduce_mean(tf.abs(y_Last-outputs_Last))
	MAE = tf.abs(y_Last-outputs_Last)
	#optimizer optimzes loss
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	training_op = optimizer.minimize(loss)

	init = tf.global_variables_initializer()
	saver = tf.train.Saver()

	with tf.Session() as sess:
		init.run()
		for epoch in range(n_epochs):
			#Commented Shuffling because we want to reserve the order for now
			#X_ , Y_,Trainseq_length = unison_shuffled_copies_Correct(X_,Y_ ,Trainseq_length )

			sess.run(training_op, feed_dict={X: X_, y: Y_  ,  seq_length:Trainseq_length ,keep_prob:trainKeepProb})
			acc_train = accuracy.eval(feed_dict={X: X_, y: Y_  ,  seq_length:Trainseq_length ,keep_prob:1.0})
			y_Last_ = y_Last.eval(feed_dict={X: X_, y: Y_  ,  seq_length:Trainseq_length ,keep_prob:1.0})
			outputs_Last_ = outputs_Last.eval(feed_dict={X: X_, y: Y_  ,  seq_length:Trainseq_length ,keep_prob:1.0})
			print("Epoch", epoch, "Train MAE =", '{0:.20f}'.format(acc_train) )
			# added by Zhe
		# 	print(outputs.get_shape())
		# cols=[]
		# hidden_states = outputs.eval(feed_dict={X: X_, y: Y_  ,  seq_length:Trainseq_length ,keep_prob:1.0})[:, :, 0]
		# num_steps = hidden_states.shape[1]

		# for i in range(1,num_steps + 1):
		# 	cols.append("steps_" +  str(i))
		# print(type(hidden_states))
		# # print(outputs.eval(feed_dict={X: X_, y: Y_  ,  seq_length:Trainseq_length ,keep_prob:1.0}).shape)
		# df = DataFrame(hidden_states,columns=cols)
		# df.to_csv(Loc+OutputFile+'.csv',index=False)
		#################

		saver.save(sess, Model_Loc+OutputFile +"LSTM_model")
		if testing:
			Output = np.zeros((TestData.shape[0] , 24))
			# added by Zhe
			# there are two cell states, we save both of them separately
			cell_states_1 = np.zeros((TestData.shape[0] , 24))
			cell_states_2 = np.zeros((TestData.shape[0] , 24))

			for i in range(TestData.shape[0]):
				seq = np.array([Testseq_length[i]])
				for j in range (24):
					X_batch = TestData[i].reshape(1, n_steps, n_inputs)
					y_pred = sess.run(outputs_Last, feed_dict={X: X_batch,seq_length:seq , keep_prob:1.0})
					Output[i,j] = y_pred.flatten()[0]
					np.roll(TestData[i], -1, axis=0)
					TestData[i] =np.roll(TestData[i], -1, axis=0)
					TestData[i , -1,0] = Output[i,j]
					if(hasMonth):
						TestData[i , -1,1:13] = getNextMonth(TestData[i , -2,1:13])
						if(Bias==None):
							TestData[i , -1,13:] = TestData[i , -2,13:]
						else:
							b_t = n_steps/(j+n_steps)
							TestData[i , -1,13:] =  b_t*TestData[i , -2,13:]+ (1-b_t)*Bias[1:]

					#################
				# added by Zhe
					cell_states_1[i, j] = states[0][0].eval(feed_dict={X: X_batch,seq_length:seq , keep_prob:1.0})[0, j]
					cell_states_2[i, j] = states[1][0].eval(feed_dict={X: X_batch,seq_length:seq , keep_prob:1.0})[0, j]

				# cols=[]
				# for i in range(1,25):
				# 	cols.append("M_"+  str(i))
				# df = DataFrame(Output,columns=cols)
				# df.to_csv(Loc+OutputFile+'.csv',index=False)
				
				cols=[]
				for i in range(1,25):
					cols.append("steps_" +  str(i))
				df_1 = DataFrame(cell_states_1, columns=cols)
				df_2 = DataFrame(cell_states_2, columns=cols)
				df_1.to_csv(Loc+OutputFile+'_cell_1_testing.csv',index=False)
				df_1.to_csv(Loc+OutputFile+'_cell_2_testing.csv',index=False)
				print("Done!")
				#################


