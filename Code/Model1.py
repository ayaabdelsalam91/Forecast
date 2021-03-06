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


Loc = '../Data/'
Model_Loc = '../Models/'



def getLSTMData(TrainInputFile , TestInputFile,numberOfFeatures , timeSteps,
	isTargetReplication = True , hasID=True):
	Traindata = Loc+TrainInputFile+'.csv'
	Traindataframe = read_csv(Traindata, names=None)
	TrainInfo = Traindataframe.values
	Testdata = Loc+TestInputFile+'.csv'
	Testdataframe = read_csv(Testdata, names=None)
	TestInfo =  Testdataframe.values




	######################### For Training ####################

	# np.random.shuffle(TrainInfo)

	TrainOutput = TrainInfo[:,-1]

	if  hasID:
		TrainData = TrainInfo[:,1:-1]
	else:
		TrainData = TrainInfo[:,:-1]

	Trainseq_length=[]
	toConsider=[]
	for i in range(TrainData.shape[0]):
		seq=0
		# print TrainData[i,:]
		for j in range (0,TrainData.shape[1],numberOfFeatures):
			if(TrainData[i,j]!=0):
				seq+=1
		if(seq>0):
			toConsider.append(i)
		Trainseq_length.append(seq)
	Trainseq_length = np.array(Trainseq_length)

	TrainOutput=TrainOutput[toConsider]
	# print (TrainOutput)
	Trainseq_length=Trainseq_length[toConsider]
	TrainData=TrainData[toConsider]


	CompleteTrainOutput = np.zeros((TrainOutput.shape[0],timeSteps,1))
	
	if isTargetReplication:
		for i in range (TrainOutput.shape[0]):
			for j in range(Trainseq_length[i]):
				CompleteTrainOutput[i,j,0] = TrainOutput[i]
	else:
		for i in range (TrainOutput.shape[0]):
			for j in range(Trainseq_length[i]):

				if(j+1 == Trainseq_length[i]):
					# if(i<10):
					# 	print("in")
					CompleteTrainOutput[i,j,0] = TrainOutput[i]
				else:
					CompleteTrainOutput[i,j,0] = TrainInfo [i ,(j+1)*numberOfFeatures]
			# if(i<10):
			# 	print ( i , TrainData[i,:(Trainseq_length[i])*4] , CompleteTrainOutput[i,:Trainseq_length[i],0]  ,  TrainOutput[i])

	NewTrainData =  TrainData.reshape((TrainData.shape[0], timeSteps ,numberOfFeatures))
	# newCompleteTrainOutput = CompleteTrainOutput.reshape(CompleteTrainOutput.shape[0],CompleteTrainOutput.shape[1])
	# df = DataFrame(newCompleteTrainOutput)
	# df.to_csv(Loc+"newCompleteTrainOutput"+'.csv',index=False)



	######################### For Testing ####################
	if  hasID:
		TestData = TestInfo[:,1:-numberOfFeatures]
	else:
		TestData = TestInfo
		
	Testseq_length=[]
	for i in range(TestData.shape[0]):
		seq=0
		for j in range (0,TestData.shape[1],numberOfFeatures):
			if(TestData[i,j]!=0):
				seq+=1
		Testseq_length.append(seq)
	Testseq_length = np.array(Testseq_length)



	NewTestData =  TestData.reshape((TestData.shape[0], timeSteps ,numberOfFeatures))

	# print( NewTrainData.shape , CompleteTrainOutput.shape , Trainseq_length.shape , NewTestData.shape , Testseq_length.shape)
	if(hasID):
		RID =TestInfo[:,0]
	else:
		RID = None
	return NewTrainData , CompleteTrainOutput , Trainseq_length ,  NewTestData, Testseq_length , RID
	


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



def getBeta(start ,current ,  number , DS=1):
	if(DS==1):
		beta =  1
	else:
		
		beta = (current -start)/number
		# print(start ,current ,  number , beta)
	return beta


def LSTM(OutputFile  , X_, Y_,Trainseq_length, TestData , Testseq_length ,
	learning_rate ,n_neurons, n_layers , alpha,n_epochs,  trainKeepProb = 1.0 ,isBaseline =None , Bias1=None , Bias2=None , RID=None  , DS = 1 ,  testing=True):
	print("Model1")

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
			# # print (X_.shape,Y_.shape , Trainseq_length.shape)
			# X_, Y_ = unison_shuffled_copies(X_,Y_)

			X_ , Y_,Trainseq_length = unison_shuffled_copies_Correct(X_,Y_ ,Trainseq_length )

			sess.run(training_op, feed_dict={X: X_, y: Y_  ,  seq_length:Trainseq_length ,keep_prob:trainKeepProb})
			acc_train = accuracy.eval(feed_dict={X: X_, y: Y_  ,  seq_length:Trainseq_length ,keep_prob:1.0})
			y_Last_ = y_Last.eval(feed_dict={X: X_, y: Y_  ,  seq_length:Trainseq_length ,keep_prob:1.0})
			# loss_ = loss.eval(feed_dict={X: X_, y: Y_  ,  seq_length:Trainseq_length ,keep_prob:1.0})
			outputs_Last_ = outputs_Last.eval(feed_dict={X: X_, y: Y_  ,  seq_length:Trainseq_length ,keep_prob:1.0})
			# printstuff(y_Last_ ,  outputs_Last_)
			if DS ==1:
				MAE_values = MAE.eval(feed_dict={X: X_, y: Y_  ,  seq_length:Trainseq_length ,keep_prob:1.0})
				interval_25 , interval_75 = get_interval(MAE_values)

			print("Epoch", epoch, "Train MAE =", '{0:.20f}'.format(acc_train) )
		saver.save(sess, Model_Loc+OutputFile +"LSTM_model")
		if testing:
			if DS ==1:
				Output = np.zeros((TestData.shape[0]*50, 2))
				for i in range(TestData.shape[0]):
						seq = np.array([Testseq_length[i]])
						for j in range(50):
							X_batch = TestData[i].reshape(1, n_steps, n_inputs)
							y_pred = sess.run(outputs_Last, feed_dict={X: X_batch , seq_length:seq ,keep_prob:1.0})
							# print (i , j , y_pred.flatten()[0] , interval_25 , interval_75)
							Output[i*50+j,0] = RID[i]
							Output[i*50+j,1] =  y_pred.flatten()[0]
							if( (y_pred.flatten()[0] - interval_25) >0):
								Output[i*50+j,2] =  y_pred.flatten()[0] - interval_25
							else:
								Output[i*50+j,2] =  y_pred.flatten()[0]
							Output[i*50+j,3] =  y_pred.flatten()[0] + interval_75

							if(seq[0]<n_steps):
								TestData[i ,seq[0],0] =Output[i*50+j,1]
								for ind in range(1,n_inputs):
									TestData[i , seq[0],ind] = TestData[i ,seq[0]-1,ind]
								seq[0]+=1

							else:
								np.roll(TestData[i], -1, axis=0)
								TestData[i] =np.roll(TestData[i], -1, axis=0)
								TestData[i , -1,0] = Output[i*50+j,1]
								if(DefaultValues==None):
									for ind in range(1,n_inputs):
											# print(TestData[i , -1,ind])
											TestData[i , -1,ind] = TestData[i ,-2,ind]
								else:
									# print("In ELSE")
									for ind in range(1,n_inputs):
											if(isBaseline[ind]==1):
												TestData[i , -1,ind] = TestData[i ,-2,ind]
											else:
												TestData[i ,-1,ind] =  DefaultValues[ind]
				df = DataFrame(Output,columns=["RID" , TargetName , "-25" , "+75"])
				df.to_csv(Loc+'.csv',index=False)
			elif DS==2:
				orignal_shape  =  TestData.shape
				TestData =  TestData.reshape(TestData.shape[0],TestData.shape[1]*TestData.shape[2])
				df = DataFrame(TestData)
				temploc = Loc+'tempTestFile.csv'
				df.to_csv(temploc,index=False)
				df_B1 = read_csv(temploc, names=None)
				df_B2 = read_csv(temploc, names=None)
				TestData_b1=df_B1.values
				TestData_b2 = df_B2.values
				TestData_b1 =  TestData_b1.reshape(orignal_shape)
				TestData_b2 =  TestData_b2.reshape(orignal_shape)
				for i in range(TestData.shape[0]):
				# for i in range(10):
					seq = np.array([Testseq_length[i]])
					startTimeStep = seq[0]
					print(i ,  Testseq_length[i])
					Data = TestData[i].reshape(n_steps, n_inputs)
					LastValue = Data[seq[0]-1,:]
					# print(LastValue)
					for j in range (129-Testseq_length[i]):

						X_batch_B1 = TestData_b1[i].reshape(1, n_steps, n_inputs)
						y_pred_B1 = sess.run(outputs_Last, feed_dict={X: X_batch_B1 , seq_length:seq ,keep_prob:1.0})

						X_batch_B2 = TestData_b2[i].reshape(1, n_steps, n_inputs)
						y_pred_B2 = sess.run(outputs_Last, feed_dict={X: X_batch_B2 , seq_length:seq ,keep_prob:1.0})
						DefaultValues_B1 = Bias1
						DefaultValues_B2 = getCenter(X_batch_B2[:,:seq[0],:],Bias2)

						beta = getBeta(startTimeStep , seq[0],n_steps,DS )

						TestData_b1[i , seq[0],0] =  y_pred_B1.flatten()[0]
						TestData_b2[i , seq[0],0] =  y_pred_B2.flatten()[0]

						for ind in range(1,n_inputs):
							if(isBaseline[ind]==1):
								TestData_b1[i , seq[0],ind] = TestData_b1[i ,seq[0]-1,ind]
								TestData_b2[i , seq[0],ind] = TestData_b2[i ,seq[0]-1,ind]
							else:
								#print (ind ,  DefaultValues_B1 ,DefaultValues_B2 ,LastValue[0,ind] , beta )
								#TestData[i ,seq[0],ind] =  beta*DefaultValues[ind] + (1-beta)*TestData[i ,seq[0]-1,ind]
								TestData_b1[i ,seq[0],ind] =  beta*DefaultValues_B1[ind] + (1-beta)*LastValue[ind]
								# print ("one" , TestData_b1[i ,seq[0],ind] , beta*DefaultValues_B1[ind] + (1-beta)*LastValue[ind] )
								TestData_b2[i ,seq[0],ind] =  beta*DefaultValues_B2[ind] + (1-beta)*LastValue[ind]
								# print("two" , TestData_b2[i ,seq[0],ind] , beta*DefaultValues_B2[ind] + (1-beta)*LastValue[ind] )
						# print("compare" ,  TestData_b1[i ,seq[0],:] ,TestData_b2[i ,seq[0],:]  )
						seq[0]+=1
				
				Output_B1 = np.zeros((TestData_b1.shape[0] , TestData_b1.shape[1]*TestData_b1.shape[2]))
				Output_B2 = np.zeros((TestData_b2.shape[0] , TestData_b2.shape[1]*TestData_b2.shape[2]))
				colindex=0
				# print(Output_B1.shape  , Output_B2.shape)
				for i in range(TestData_b1.shape[0]):
					for j in range (TestData_b1.shape[1]):
						# print(i,j,TestData[i,j,:])
						Output_B1[i,colindex:colindex+ TestData_b1.shape[2]] = TestData_b1[i,j,:]
						Output_B2[i,colindex:colindex+ TestData_b2.shape[2]] = TestData_b2[i,j,:]
						colindex+=TestData_b1.shape[2]
					colindex=0
				df_B1 = DataFrame(Output_B1)
				df_B1.to_csv(Loc+OutputFile+'_B1.csv',index=False)
				df_B2 = DataFrame(Output_B2)
				df_B2.to_csv(Loc+OutputFile+'_B2.csv',index=False)



