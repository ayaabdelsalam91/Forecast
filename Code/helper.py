import sklearn.metrics as sm
import numpy as np
from pandas import read_csv, DataFrame, to_datetime
import os
import glob
import csv
import random

Loc = '../Data/'


def getNormializedData(inputfile ,OutputFile,FeatureIndex,flag=True ,maxV=None , minV=None):
	data = Loc+inputfile+ '.csv'
	dataframe = read_csv(data, names=None)
	dataframeCols = dataframe.columns.values
	info=dataframe.values
	Output =info 
	if flag:
		maxV = np.zeros((1,info.shape[1]))
		minV = np.zeros((1,info.shape[1]))
		for i in range(FeatureIndex,info.shape[1]):
			# print(info[:,i])
			maxV[0,i] = np.amax(info[:,i])
			minV[0,i] = np.amin(info[:,i])
	# print (maxV,minV)
	for i in range (info.shape[0]):
		for j in range(FeatureIndex,info.shape[1]):
			Output[i,j] = (Output[i,j]-minV[0,j])/(maxV[0,j]-minV[0,j])
	df=DataFrame(Output, columns=dataframeCols)
	df.to_csv(Loc+OutputFile+'.csv',index=False)
	return maxV , minV


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



