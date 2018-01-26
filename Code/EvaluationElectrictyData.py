import sklearn.metrics as sm
import numpy as np
from pandas import read_csv, DataFrame, to_datetime
import os
import glob
import csv
import random
from helper import getDenormalizedData
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

Loc = '../Data/'
Loc_Graph = '../Graphs/'
def getMAEForElectricty(NN_Input , NN_Output , Actual_Values ,numberOfFeatures , Denormalize=False , DenormalizeFile =None ,  FeatureIndex = None):
	data = Loc+NN_Input+ '.csv'
	dataframe = read_csv(data, names=None)
	NN_InputCols = dataframe.columns.values
	NN_InputInfo = dataframe.values
	data = Loc+NN_Output+ '.csv'
	dataframe = read_csv(data, names=None)
	NN_OutputCols = dataframe.columns.values
	NN_OutputInfo = dataframe.values

	data = Loc+Actual_Values+ '.csv'
	dataframe = read_csv(data, names=None)
	Actual_ValuesCols = dataframe.columns.values
	Actual_ValuesInfo = dataframe.values
	NumberOfMonth = int(NN_InputInfo.shape[1]/numberOfFeatures)

	MonthError = np.zeros((NumberOfMonth,2))

	lastValue = np.zeros((NN_InputInfo.shape[0],1))
	for i in range(NN_InputInfo.shape[0]):
		for j in range  (NN_InputInfo.shape[1]):
			if(NN_InputInfo[i][j]!=0):
				lastValue[i]=j
			else:
				break

	# for i in range(NN_InputInfo.shape[0]):
	# 	print(int(lastValue[i]/numberOfFeatures))
	newData = np.abs(NN_OutputInfo - Actual_ValuesInfo)
	error = 0
	count = 0
	for j in range(NN_OutputInfo.shape[1]):
		if("F_1" in NN_InputCols[j]):
			for i in range (NN_OutputInfo.shape[0]):
				if NN_InputInfo[i][j] == 0:
					error+=newData[i][j]
					count+=1
					#print  (i , j , newData[i][j] , lastValue[i] ,int((j-lastValue[i])/numberOfFeatures) )
					MonthError[int((j-lastValue[i])/numberOfFeatures)][0]+=newData[i][j]
					MonthError[int((j-lastValue[i])/numberOfFeatures)][1]+=1
	Month = []
	for i in range (MonthError.shape[0]-1):
		#print(i , MonthError[i,0] ,MonthError[i,1] ,MonthError[i,0]/MonthError[i,1])
		Month.append(i)
	print("MAE = "  ,  error/count ,  error, count)
	MAEperMonth=  MonthError[:,0]/MonthError[:,1]
	print(MAEperMonth.shape)
	MAEperMonth = MAEperMonth[:-1]
	print( MAEperMonth.shape)
	if(Denormalize):
		DenormalizedMAEperMonth = getDenormalizedData(DenormalizeFile ,MAEperMonth,FeatureIndex)
		plt.plot(Month,DenormalizedMAEperMonth)
	else:
		plt.plot(Month,MAEperMonth)
	# plt.axis([0, 6, 0, 20])
	plt.xlabel('Month')
	plt.ylabel('Megawatthours MAE')
	plt.title('MAE for Long-horizon Forecast')
	plt.savefig(Loc_Graph+NN_Output+'.png')
	plt.show()
	if(Denormalize):
		return Month , DenormalizedMAEperMonth
	else:
		return Month, MAEperMonth


			
# Month ,B2 = getMAEForElectricty( 'TestingElectrictyDS','EDSTestOuputM1___B2' ,'TestingElectrictyDSAns',4 , Denormalize=True , DenormalizeFile='TrainingElectrictyDSReOrdered',FeatureIndex=3)

# Month , B1 = getMAEForElectricty( 'TestingElectrictyDS','EDSTestOuputM1___B1' ,'TestingElectrictyDSAns',4 , Denormalize=True , DenormalizeFile='TrainingElectrictyDSReOrdered',FeatureIndex=3)

Month , Baseline =  getMAEForElectricty( 'TestingElectrictyDS_Uni','EDSTestOuputBaseline_' ,'TestingElectrictyDSAns_Uni',1 , Denormalize=True , DenormalizeFile='TrainingElectrictyDSReOrdered',FeatureIndex=3)




plt.plot(Month, B1, Month, B2 ,  Month, Baseline)
plt.show()