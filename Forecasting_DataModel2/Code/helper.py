import sklearn.metrics as sm
import numpy as np
from pandas import read_csv, DataFrame, to_datetime
import os
import glob
import csv
import random

Loc = '../Data/'




def getNormializedData(inputfile ,OutputFile,FeatureIndex,flag=True ,mean=None , variance=None  , AddMonthFlag=False, AddStateFlag=False):
	data = Loc+inputfile+ '.csv'
	dataframe = read_csv(data, names=None)
	dataframeCols = dataframe.columns.values
	info=dataframe.values
	Month = dataframe.columns.get_loc("Month")
	Output =info 
	OneHotMonth =  np.zeros((Output.shape[0] , 12))
	if(AddStateFlag):
		State = dataframe.columns.get_loc("State")
		OneHotState =  np.zeros((Output.shape[0] , 51))
		Statedataframe = read_csv(StateFile, names=None)
		stateInfo=Statedataframe.values
		state_dict = {}
		for i in range(stateInfo.shape[0]):
			state_dict[stateInfo[i,0]] = stateInfo[i,1]
		for i in range (info.shape[0]):

			OneHotState[i,state_dict[info[i,State]]]=1


	if flag:
		mean = np.zeros((1,info.shape[1]))
		variance = np.zeros((1,info.shape[1]))
		for i in range(FeatureIndex,info.shape[1]):
			# print(info[:,i])
			mean[0,i] = np.mean(info[:,i])
			variance[0,i] = np.var(info[:,i])
	# print (maxV,minV)
	for i in range (info.shape[0]):
		for j in range(FeatureIndex,info.shape[1]):
			Output[i,j] = (Output[i,j]-mean[0,j])/(variance[0,j])

		OneHotMonth[i ,Output[i,Month]-1] =1 
		# print(Output[i,Month] , OneHotMonth[i , Output[i,Month]-1])

	ExpandedOutput = np.hstack(( Output[:,:FeatureIndex] ,OneHotMonth ,Output[:,FeatureIndex:] ))
	if(AddStateFlag):
		ExpandedOutput = np.hstack(( Output[:,:FeatureIndex] ,OneHotMonth , OneHotState ,Output[:,FeatureIndex:] ))
	cols = []
	for i in range(FeatureIndex):
		cols.append(dataframeCols[i])
	for i in range(1,13):
		cols.append("M_"+ str(i))
	if(AddStateFlag):
		for i in range(1,52):
			cols.append("State_"+ str(i))
	for i in range(FeatureIndex , info.shape[1]):
		cols.append(dataframeCols[i])

	if( AddMonthFlag):
		df=DataFrame(ExpandedOutput, columns=cols)
		df.to_csv(Loc+OutputFile+'.csv',index=False)
	else:
		df=DataFrame(Output, columns=dataframeCols)
		df.to_csv(Loc+OutputFile+'.csv',index=False)
	return mean , variance


def getDenormalizedData(MinMaxFile , dataToBeDenormalized ,  FeatureIndexOfIntrest):
	data = Loc+MinMaxFile+ '.csv'
	dataframe = read_csv(data, names=None)
	dataframeCols = dataframe.columns.values
	info=dataframe.values
	maxV = np.amax(info[:,FeatureIndexOfIntrest])
	minV = np.amin(info[:,FeatureIndexOfIntrest])
	diff = maxV-minV
	newdata = np.zeros((dataToBeDenormalized.shape))
	for i in range(dataToBeDenormalized.shape[0]):
		newdata[i] = (dataToBeDenormalized[i]*diff) + minV

	return newdata



def getSubsetOfData (inputfile ,OutputFile,rows=None,NumberOfSamples=120):
	data = Loc+inputfile+ '.csv'
	dataframe = read_csv(data, names=None)
	dataframeCols = dataframe.columns.values
	info=dataframe.values
	if rows == None:
		rows = random.sample(range(0, info.shape[0]), NumberOfSamples)
	newInfo = info[rows,:]

	df=DataFrame(newInfo, columns=dataframeCols)
	df.to_csv(Loc+OutputFile+'.csv',index=False)
	print (rows)
	return rows


def isTheSame(file1 ,  file2):
	data = Loc+file1+ '.csv'
	dataframe = read_csv(data, names=None)
	dataframeCols = dataframe.columns.values
	info1=dataframe.values
	data = Loc+file2+ '.csv'
	dataframe = read_csv(data, names=None)
	dataframeCols = dataframe.columns.values
	info2=dataframe.values
	if(info2.shape!=info1.shape):
		return False
	for i in range(info2.shape[0]):
		for j in range(info2.shape[1]):
			if(info1[i,j]!=info2[i,j]):
				return False
	return True

	
def getNextMonth(OneHotVector):
	NewOneHot = np.zeros((OneHotVector.shape))
	Index = np.argmax(OneHotVector)
	# print(OneHotVector)
	# print (Index)
	if(Index == OneHotVector.shape[0]-1):
		Index=0
	else:
		Index+=1
	NewOneHot[Index]=1
	return NewOneHot


