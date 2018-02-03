import sklearn.metrics as sm
import numpy as np
from pandas import read_csv, DataFrame, to_datetime
import os
import glob
import csv
import random
from helper import getDenormalizedData ,isTheSame
import matplotlib.pyplot as plt
import datetime
from datetime import date
from dateutil.relativedelta import relativedelta
np.set_printoptions(suppress=True)


Loc = '../Data/'
Loc_Graph = '../Graphs/'
def getMAEForElectricty(NN_Input , NN_Output , Actual_Values ,numberOfFeatures , Denormalize=False , DenormalizeFile =None ,  FeatureIndex = None):
	print(NN_Input)
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
	# print(MAEperMonth.shape)
	MAEperMonth = MAEperMonth[:-1]
	# print( MAEperMonth.shape)
	if(Denormalize):
		DenormalizedMAEperMonth = getDenormalizedData(DenormalizeFile ,MAEperMonth,FeatureIndex)
		# plt.plot(Month,DenormalizedMAEperMonth)
	# else:
		# plt.plot(Month,MAEperMonth)
	# plt.axis([0, 6, 0, 20])
	# plt.xlabel('Month')
	# plt.ylabel('Megawatthours MAE')
	# plt.title('MAE for Long-horizon Forecast')
	# plt.savefig(Loc_Graph+NN_Output+'.png')
	# plt.show()
	if(Denormalize):
		#print(DenormalizedMAEperMonth)
		
		print("hereeee" , np.nansum(DenormalizedMAEperMonth))
		return Month , DenormalizedMAEperMonth
	else:
		print("NOT HERE"  , np.nansum(MAEperMonth))
		return Month, MAEperMonth
		

def getMAEForElectricty(NN_Output , Actual_Values,hasIndex=False , Denormalize=False , DenormalizeFile =None , FeatureIndex=3 ):
	data = Loc+NN_Output+ '.csv'
	dataframe = read_csv(data, names=None)
	NN_OutputCols = dataframe.columns.values
	NN_OutputInfo = dataframe.values

	data = Loc+Actual_Values+ '.csv'
	dataframe = read_csv(data, names=None)
	Actual_ValuesCols = dataframe.columns.values
	Actual_ValuesInfo = dataframe.values
	print(NN_OutputInfo.shape , Actual_ValuesInfo.shape )
	if(hasIndex):
		Actual_ValuesInfo = Actual_ValuesInfo[:,1:]
	print(NN_OutputInfo.shape , Actual_ValuesInfo.shape )
	NumberOfMonth =Actual_ValuesInfo.shape[1]

	newData = np.abs(Actual_ValuesInfo - NN_OutputInfo  )
	MAEperMonth = np.zeros((NumberOfMonth,1))
	Month = []
	for i in range(NumberOfMonth):
		MAEperMonth[i] = np.mean(newData[:,i])
		Month.append(i)
		print(i,MAEperMonth[i] )
	print(newData.shape)
	max=0
	maxIndex=0
	fig, ax = plt.subplots()
	for i in range(newData.shape[0]):
		# print(i ,  np.sum(newData[i,:]))
		ax.plot(Month, newData[i,:])
		if( np.sum(newData[i,:])>max):
			max =  np.sum(newData[i,:])
			maxIndex=i

	legend = ax.legend()
	plt.xlabel('Month')
	plt.ylabel('MAE')
	plt.savefig(Loc_Graph+NN_Output+"_ErrorAll"+'.png')
	plt.show()
	print(maxIndex , max)

	if(Denormalize):
		DenormalizedMAEperMonth = getDenormalizedData(DenormalizeFile ,MAEperMonth,FeatureIndex)

		return Month , DenormalizedMAEperMonth
	else:

		return Month, MAEperMonth



def PlotTS(Infile):
	data = Loc+Infile+ '.csv'
	dataframe = read_csv(data, names=None)
	infoCols= dataframe.columns.values
	info = dataframe.values
	# Month = np.array([datetime.datetime(2007,1, i) for i in range(72)])
	# Month = np.arange(datetime.datetime(2007,1, 1), datetime.datetime(2017,10,1),relativedelta(months=1)).astype(datetime)
	# print(Month)
	base = date(2007,1, 1)
	Month = [base]
	for i in range(129):
		# print(	base)
		base = base +relativedelta(months=+1)
		Month.append(base)
	Map = np.zeros((info.shape))
	for i in range(info.shape[0]):
		for j in range(1 , info.shape[1]):
			if (info[i,j]>info[i,j-1]):
				Map[i,j]=1
			elif (info[i,j]<info[i,j-1]):
				Map[i,j]=-1
	df = DataFrame(Map,columns=Month)
	df.to_csv(Loc+'Map'+'.csv',index=False)

	fig, ax = plt.subplots()
	for i in range(info.shape[0]):
		ax.plot(Month, info[i,:])


	legend = ax.legend()
	plt.xlabel('Month')



	plt.ylabel('Megawatthours')
	plt.savefig(Loc_Graph+"EntireTS"+'.png')

	plt.show()
	Newinfo = np.mean(info,axis=0)
	# for i in range (Newinfo.shape[0]):
	# 	print (i, Newinfo[i])
	print (Newinfo.shape)
	return Month , Newinfo

Month ,Baseline = getMAEForElectricty('BL_500' ,'TestingAnsDS')
Month ,b2 = getMAEForElectricty('B2_500_' ,'TestingAnsDS')
Month ,b3 = getMAEForElectricty('B3_500_' ,'TestingAnsDS')
Month ,b4 = getMAEForElectricty('B4_500_' ,'TestingAnsDS')
# Month ,b5 = getMAEForElectricty('B2_Multi_1000_NormalizedK5_Corrected' ,'TestingAnsDS',hasIndex=True ,FeatureIndex=3)

fig, ax = plt.subplots()
ax.plot(Month, Baseline,'g',label='Baseline')
ax.plot(Month, b2 ,'r',label='k=2')
ax.plot(Month, b3,'b',label='K=3')
ax.plot(Month,b4 ,'m',label= 'K=4')
# ax.plot(Month, b5,'y' ,label='K=5')
legend = ax.legend()


# fig, ax = plt.subplots()
# # ax.plot(Month, B1, 'r', label='Bias 1')
# ax.plot(Month, B2, 'g', label='Cluster Bias ')
# ax.plot(Month, Baseline, 'b', label='Basline')

# legend = ax.legend()


# B2_Multi_1000

# Month ,  MeanTS = PlotTS('electricityDSTS_uni')
# plt.plot( Month, MeanTS)
# plt.xlabel('Month')

# plt.ylabel('Megawatthours')
# plt.savefig(Loc_Graph+"TS"+'.png')
# plt.show()
#plt.plot( Month, b2,'g')
plt.xlabel('Month')
plt.ylabel('MAE')
plt.title('Second Test Scheme ')
plt.savefig(Loc_Graph+"2ndAll"+'.png')
plt.show()

