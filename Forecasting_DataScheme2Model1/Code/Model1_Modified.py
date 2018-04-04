from pandas import DataFrame
from pandas import read_csv
import random
import math
import numpy as np 
import tensorflow  as tf
from numpy import zeros, newaxis
from sklearn.model_selection import train_test_split
from scipy.ndimage.interpolation import shift
from Bias import getCenter , getOneCenter_
np.set_printoptions(suppress=True)
from helper import getNextMonth


Loc = '../Data/'
Model_Loc = '../Models/'





def getLSTMData(TrainInputFile , TestInputFile,numberOfFeatures , timeSteps,
	isTargetReplication = True , hasID=True ,  model=None ,  Bias=None ,hasMonth=False):
	Traindata = Loc+TrainInputFile+'.csv'
	Traindataframe = read_csv(Traindata, names=None)
	TrainInfo = Traindataframe.values
	Testdata = Loc+TestInputFile+'.csv'
	Testdataframe = read_csv(Testdata, names=None)
	TestInfo =  Testdataframe.values
	TestInfoCols = Testdataframe.columns.values

 ######################## For Training ####################

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

	######################### For Testing ####################
	if  hasID :
		TestData = TestInfo[:,1:-numberOfFeatures]
	elif  "State" in TestInfoCols[0] :
		TestData = TestInfo[:,1:]
	else:
		TestData = TestInfo
	
	# print(numberOfFeatures)
	Testseq_length=[]
	for i in range(TestData.shape[0]):
		seq=0
		for j in range (0,TestData.shape[1],numberOfFeatures):
			if(TestData[i,j]!=0):
				seq+=1
		# print(TestData[i,:] ,seq)
		Testseq_length.append(seq)
	Testseq_length = np.array(Testseq_length)

	oneSum=0
	zerosSum=0
	if(model==None):
		NewTrainData =  TrainData.reshape((TrainData.shape[0], timeSteps ,numberOfFeatures))
		NewTestData =  TestData.reshape((TestData.shape[0], timeSteps ,numberOfFeatures))


	elif(model!=None):
		TempTrainData =  TrainData.reshape((TrainData.shape[0], timeSteps ,numberOfFeatures))
		if(not hasMonth):
			NewTrainData = np.zeros(((TrainData.shape[0], timeSteps ,numberOfFeatures*2)))
			for i in range (TempTrainData.shape[0]):
				for j in range (TempTrainData.shape[1]):
					if(np.sum(TempTrainData[i,j,:])!=0):
						NewTrainData[i,j,:numberOfFeatures] = TempTrainData[i,j,:]
						center ,index = getOneCenter_(TempTrainData[i,j,:],model)
						if(index==1):
							oneSum+=1
						else:
							zerosSum+=1
						NewTrainData[i,j,numberOfFeatures:]=center

			TempTestData =  TestData.reshape((TestData.shape[0], timeSteps ,numberOfFeatures))
			NewTestData = np.zeros(((TestData.shape[0], timeSteps ,numberOfFeatures*2)))
			for i in range (TempTestData.shape[0]):
				for j in range (TempTestData.shape[1]):
					if(np.sum(TempTestData[i,j,:])!=0):
						NewTestData[i,j,:numberOfFeatures] = TempTestData[i,j,:]
						center , index = getOneCenter_(TempTestData[i,j,:],model)
						if(index==1):
							oneSum+=1
						else:
							zerosSum+=1
						NewTestData[i,j,numberOfFeatures:]=center
		else:
			newNumberOfFeatures = (numberOfFeatures-12)*2+12
			NewTrainData = np.zeros(((TrainData.shape[0], timeSteps ,newNumberOfFeatures)))
			print(NewTrainData.shape)
			for i in range (TempTrainData.shape[0]):
				for j in range (TempTrainData.shape[1]):
					if(np.sum(TempTrainData[i,j,:])!=0):
						NewTrainData[i,j,:numberOfFeatures] = TempTrainData[i,j,:]
						TempData = np.zeros((numberOfFeatures-12))
						TempData[0] = TempTrainData[i,j,0]
						TempData[1:] =  TempTrainData[i,j,13:]
						# print(TempTrainData[i,j,:])
						# print(TempData)
						# print("")
						center ,index = getOneCenter_(TempData,model)
						if(index==1):
							oneSum+=1
						else:
							zerosSum+=1
						NewTrainData[i,j,numberOfFeatures:]=center
			print(TestData.shape)
			TempTestData =  TestData.reshape((TestData.shape[0], timeSteps ,numberOfFeatures))
			NewTestData = np.zeros(((TestData.shape[0], timeSteps ,newNumberOfFeatures)))
			for i in range (TempTestData.shape[0]):
				for j in range (TempTestData.shape[1]):
					if(np.sum(TempTestData[i,j,:])!=0):
						NewTestData[i,j,:numberOfFeatures] = TempTestData[i,j,:]
						TempData = np.zeros((numberOfFeatures-12))
						TempData[0] = TempTestData[i,j,0]
						TempData[1:] =  TempTestData[i,j,13:]
						# print(TempTrainData[i,j,:])
						# print(TempData)
						# print("")
						center , index = getOneCenter_(TempData,model)
						if(index==1):
							oneSum+=1
						else:
							zerosSum+=1
						NewTestData[i,j,numberOfFeatures:]=center
			print("oneSum" , oneSum , "zerosSum" , zerosSum)

	# elif(Bias!=None):
	# 	TempTrainData =  TrainData.reshape((TrainData.shape[0], timeSteps ,numberOfFeatures))
	# 	NewTrainData = np.zeros(((TrainData.shape[0], timeSteps ,numberOfFeatures*2)))
	# 	for i in range (TempTrainData.shape[0]):
	# 		for j in range (TempTrainData.shape[1]):
	# 			NewTrainData[i,j,:numberOfFeatures] = TempTrainData[i,j,:]
	# 			NewTrainData[i,j,numberOfFeatures:]=Bias

	# 	TempTestData =  TestData.reshape((TestData.shape[0], timeSteps ,numberOfFeatures))
	# 	NewTestData = np.zeros(((TestData.shape[0], timeSteps ,numberOfFeatures*2)))
	# 	for i in range (TempTestData.shape[0]):
	# 		for j in range (TempTestData.shape[1]):
	# 			NewTestData[i,j,:numberOfFeatures] = TempTestData[i,j,:]
	# 			NewTestData[i,j,numberOfFeatures:]=Bias

	print( NewTrainData.shape , CompleteTrainOutput.shape , Trainseq_length.shape , NewTestData.shape , Testseq_length.shape)
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
	learning_rate ,n_neurons, n_layers , alpha,n_epochs,  trainKeepProb = 1.0 ,isBaseline =None , Bias=None ,RID=None  , DS = 1 ,  testing=True,KmeanModel=None,hasMonth=False):
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
				Output = np.zeros((TestData.shape[0] , 24))
				for i in range(TestData.shape[0]):
	

				# for i in range(2):
					if(not hasMonth):
						seq = np.array([Testseq_length[i]])
						startTimeStep = seq[0]
						print(i ,  Testseq_length[i])
						Data = TestData[i].reshape(n_steps, n_inputs)
						LastValue = Data[seq[0]-1,:]
						# print(LastValue)
						for j in range (n_steps-Testseq_length[i]):

							X_batch_B1 = TestData[i].reshape(1, n_steps, n_inputs)
							y_pred_B1 = sess.run(outputs_Last, feed_dict={X: X_batch_B1 , seq_length:seq ,keep_prob:1.0})

							TestData[i , seq[0],0] =  y_pred_B1.flatten()[0]

							for ind in range(1,int(n_inputs/2)):
								TestData[i , seq[0],ind] = TestData[i ,seq[0]-1,ind]
							if(KmeanModel!=None):
								center , index = getOneCenter_(TestData[i , seq[0],:int(n_inputs/2)],KmeanModel)
							else:
								center = Bias
							TestData[i , seq[0],int(n_inputs/2):] = center
							seq[0]+=1
						NewTestData = TestData[:,:,:int(n_inputs/2)]
						Output = np.zeros((NewTestData.shape[0] , NewTestData.shape[1]*NewTestData.shape[2]))
						colindex=0
						for i in range(TestData.shape[0]):
							for j in range (TestData.shape[1]):
								Output[i,colindex:colindex+ NewTestData.shape[2]] = NewTestData[i,j,:]
								colindex+=NewTestData.shape[2]
							colindex=0
					else:
						seq = np.array([Testseq_length[i]])
						
						for j in range (24):
						#for j in range (3):
	
							X_batch = TestData[i].reshape(1, n_steps, n_inputs)
							y_pred = sess.run(outputs_Last, feed_dict={X: X_batch ,seq_length:seq  ,  keep_prob:1.0})
							Output[i,j] = y_pred.flatten()[0]
							np.roll(TestData[i], -1, axis=0)
							TestData[i] =np.roll(TestData[i], -1, axis=0)
							TestData[i , -1,0] = Output[i,j]
							TestData[i , -1,1:13] = getNextMonth(TestData[i , -2,1:13])
							if(Bias==None and KmeanModel==None):
								TestData[i , -1,13:] = TestData[i , -2,13:]
							else:
								if(KmeanModel!=None):
									DataInput = np.zeros((4))
									DataInput[0] =  Output[i,j]
									DataInput[1:] =  TestData[i , -2,13:16]
									# print ("DataInput",DataInput)
									center , index = getOneCenter_(DataInput,KmeanModel)
								else:
									center = Bias
								# print("-2" , TestData[i ,-2,:])
								# print("center" , center)
								# print("TestData[i ,-2,13:16]" , TestData[i ,-2,13:16])
								# print("TestData[i ,-2,16:]" , TestData[i ,-2,16:])
								TestData[i ,-1,13:16] = TestData[i ,-2,13:16]
								TestData[i ,-1,16:] = center
				
							# print("After"  , TestData[i , -1,1:])
							# print("")
							# print("")
							# print("")
						# print("")



			cols=[]
			for i in range(1,25):
				cols.append("M_"+  str(i))
			df = DataFrame(Output,columns=cols)
			df.to_csv(Loc+OutputFile+'.csv',index=False)





