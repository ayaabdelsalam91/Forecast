from pandas import DataFrame
from pandas import read_csv
import random
import numpy as np


Loc = '../Data/'


def getLSTMData(TrainInputFile , TestInputFile,numberOfFeatures , timeSteps,
	isTargetReplication = True , hasID=True):
	Traindata = Loc+TrainInputFile+'.csv'
	Traindataframe = read_csv(Traindata, names=None)
	TrainInfo = Traindataframe.values
	Testdata = Loc+TestInputFile+'.csv'
	Testdataframe = read_csv(Testdata, names=None)
	TestInfo =  Testdataframe.values



	######################### For Training ####################

	np.random.shuffle(TrainInfo)

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
	Trainseq_length=Trainseq_length[toConsider]
	TrainData=TrainData[toConsider]


	CompleteTrainOutput = np.zeros((TrainOutput.shape[0],timeSteps,1))
	
	if isTargetReplication:
		for i in range (TrainOutput.shape[0]):
			for j in range(Trainseq_length[i]):
				CompleteTrainOutput[i,j,0] = TrainOutput[i]

	NewTrainData =  TrainData.reshape((TrainData.shape[0], timeSteps ,numberOfFeatures))

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

	print( NewTrainData.shape , CompleteTrainOutput.shape , Trainseq_length.shape , NewTestData.shape , Testseq_length.shape)
	if(hasID):
		RID =TestInfo[:,0]
	else:
		RID = None
	return NewTrainData , CompleteTrainOutput , Trainseq_length ,  NewTestData, Testseq_length , RID
	