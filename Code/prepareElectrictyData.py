import sklearn.metrics as sm
import numpy as np
from pandas import read_csv, DataFrame, to_datetime
import os
import glob
import csv
import random

Loc = '../Data/'
data = Loc+ 'electricityDataset.csv'
Loc2 = '/Users/aya/Documents/Research/Dataset/State_temp/'
states = Loc+ '50_us_states_all_data.csv'

def UpdateTempList():
	statesdataframe = read_csv(states, names=None)
	infostates=statesdataframe.values
	AllTemp = np.zeros((132*51,4) ,dtype=infostates.dtype)
	index=0
	cols =  ['Name','Year',	'Month','Value']
	extension = 'csv'
	os.chdir(Loc2)
	files = [i for i in glob.glob('*.{}'.format(extension))]
	flag  = np.zeros((infostates.shape[0],1))
	for file in files:
		print(file)
		with open(file, newline='') as f:
			reader = csv.reader(f, delimiter=',', quotechar='"')
			row1 = next(reader) 
			for i in range(infostates.shape[0]):
				if(infostates[i][0]== row1[0]):
					print(infostates[i][1] , row1[0])
					flag[i]=1
					tempdataframe = read_csv(file, skiprows=4, header=None)
					tempinfo = tempdataframe.values
					for  j in  range(tempinfo.shape[0]):
						date = to_datetime(int(tempinfo[j,0]), format='%Y%m')
						AllTemp[index,0] = infostates[i][1]
						AllTemp[index,1] = date.year
						AllTemp[index,2] = date.month
						AllTemp[index,3] = tempinfo[j,1]
						index+=1
	df=DataFrame(AllTemp, columns=cols)
	df.to_csv(Loc+'AllTemp.csv',index=False)
	for i in range(infostates.shape[0]):
		if(flag[i]==0):
			print (infostates[i][0] ,  infostates[i][1])


def addTemp():
	dataframe = read_csv(data, names=None)
	info=dataframe.values
	newinfo = np.zeros(info.shape ,dtype=info.dtype)

	State =dataframe.columns.get_loc("State")
	Year=dataframe.columns.get_loc("Year")
	Month=dataframe.columns.get_loc("Month")
	Megawatthours =dataframe.columns.get_loc("Megawatthours")
	Count=dataframe.columns.get_loc("Count")
	price =dataframe.columns.get_loc("Cents/kWh")

	tempdataframe = read_csv(Loc+'AllTemp.csv',names=None)
	tempinfo=tempdataframe.values
	tempName =tempdataframe.columns.get_loc("Name")
	tempYear=tempdataframe.columns.get_loc("Year")
	tempMonth=tempdataframe.columns.get_loc("Month")
	tempvalue=tempdataframe.columns.get_loc("Value")


	flag = 0
	print (info.shape[0]  ,tempinfo.shape[0])

	for i in range(info.shape[0]):
		for j in range(tempinfo.shape[0]):
			print(i, j)
			if(info[i,State]==tempinfo[j,tempName] 
				and  int(info[i,Year]) == int(tempinfo[j,tempYear])
				and  int(info[i,Month]) == int(tempinfo[j,tempMonth])):
			#0		1        2       3                   4               5        6
			#Year	Month	State	Thousand Dollars	Megawatthours	Count	Cents/kWh
				newinfo[i,0] = info[i,Year]
				newinfo[i,1] = info[i,Month]
				newinfo[i,2] = info[i,State]
				newinfo[i,3] = tempinfo[j,tempvalue]
				newinfo[i,4] = info[i,Count]
				newinfo[i,5] = info[i,price]
				newinfo[i,6] = info[i,Megawatthours]
				flag=1
			else:
				flag=0
			if(flag==1):
				break
			# if(int(tempinfo[j,tempYear]) >  int(info[i,Year])):
			# 	break;


	cols = ['Year',	'Month'	 , 'State' , 	'Temp' , 'Count' 	, 'price','Megawatthours']
	df=DataFrame(newinfo, columns=cols)
	df.to_csv(Loc+'electricityDatasetWithTemp.csv',index=False)



def divideDataToTrainingAndTesting(inputfile,OuputFile):
	data = Loc+inputfile+ '.csv'
	dataframe = read_csv(data, names=None)
	info=dataframe.values
	cols = dataframe.columns.values
	State = dataframe.columns.get_loc("State")
	States = info[:,State]
	unqStates = np.unique(States)
	testingAmount = unqStates.shape[0]*0.2
	indexForTestingStates = random.sample(range(0, int(unqStates.shape[0]-1)), int(testingAmount))
	testingStates = []
	for i in range (unqStates.shape[0]):
		if i in indexForTestingStates:
			testingStates.append(unqStates[i])
	indexForTraining = []
	indexForTesting = []
	for i in range (info.shape[0]):
		if(info[i,State] in testingStates):
			indexForTesting.append(i)
		else:
			indexForTraining.append(i)
	testingData = info[indexForTesting,:]
	trainingData = info[indexForTraining,:]
	dfTesting=DataFrame(testingData, columns=cols)
	dfTesting.to_csv(Loc+OuputFile+'Testing.csv',index=False)
	dfTraining=DataFrame(trainingData, columns=cols)
	dfTraining.to_csv(Loc+OuputFile+'Training.csv',index=False)



def getTimeSeries(inputfile,OuputFile ,):
	data = Loc+inputfile+ '.csv'
	dataframe = read_csv(data, names=None)
	dataframe =dataframe.sort_values(by=['State', 'Year', 'Month'])
	dataframeCols = dataframe.columns.values
	info=dataframe.values
	State = dataframe.columns.get_loc("State")
	Temp = dataframe.columns.get_loc("Temp")
	States = info[:,State]
	unqStates = np.unique(States)
	size = int(info.shape[0]/unqStates.shape[0])
	NumberOfFeatures = int(info.shape[1]-3)
	yDim = size*NumberOfFeatures+1
	newinfo = np.zeros((unqStates.shape[0],yDim),dtype=info.dtype)

	index = 0
	current_state= info[0,State]
	newinfo[0,0]=current_state
	colIndex=1
	for i in range (info.shape[0]):
		# print (i,index,colIndex , info[i,State], current_state)
		if(info[i,State]==current_state):
			newinfo[index,colIndex:colIndex+4]=info[i,Temp:]
			colIndex+=4
		else:
			colIndex=1
			index+=1
			current_state = info[i,State]
			newinfo[index,0]=current_state

			newinfo[index,colIndex:colIndex+4]=info[i,Temp:]
			colIndex+=4


	cols = ['State']
	CurrentYear=2007
	Month=1
	Feature=3
	for i in range(1,yDim):
		if(Feature<NumberOfFeatures+3):
			cols.append(str(CurrentYear)+'_'+str(Month)+'_'+dataframeCols[Feature])
			Feature+=1
		else:
			Feature=3
			if(Month==12):
				Month=1
				CurrentYear+=1
			else:
				Month+=1
			cols.append(str(CurrentYear)+'_'+str(Month)+'_'+dataframeCols[Feature])
			Feature+=1
		
	df=DataFrame(newinfo, columns=cols)
	df.to_csv(Loc+OuputFile+'.csv',index=False)



def divideTStoModelData(inputfile,OuputFile,TargetIndex,isTesting=False,AnsFile=None):
	data = Loc+inputfile+ '.csv'
	dataframe = read_csv(data, names=None)
	dataframeCols = dataframe.columns.values
	info=dataframe.values
	if(isTesting==False):
		newinfo = np.zeros((int(info.shape[0]*129),int(info.shape[1]-4)))
	else:
		newinfo = np.zeros((int(info.shape[0]*129),int(info.shape[1]-5)))
		Ansinfo = np.zeros((int(info.shape[0]*129),int(info.shape[1]-5)))
	infoYindex = TargetIndex
	print  (infoYindex)
	newinfoXindex=0
	newinfoYindex=0
	row=0
	for i in range(info.shape[0]):

		while (infoYindex < info.shape[1]-1):
			# print(infoYindex+4 , TargetIndex  , , Ansinfo.shape )
			newinfo[newinfoXindex,:infoYindex] = info[i,TargetIndex:infoYindex+4]
			if(isTesting):
				Ansinfo [newinfoXindex] =  info[i,TargetIndex:-1]
			infoYindex+=4
			if(isTesting==False):
				newinfo[newinfoXindex , -1] = info[i,infoYindex]
			# print(newinfo[newinfoXindex , -1] )
			newinfoXindex+=1
		# print("Out")
		newinfoYindex=0
		infoYindex = TargetIndex
	month =1
	year =1
	Feature=1
	cols =[]
	CurrentYear=1
	Month=1
	for i in range(int(info.shape[1]-5)):
		if(Feature<5):
			cols.append('Y_'+str(CurrentYear)+'_M_'+str(Month)+'_F_'+str(Feature))
			Feature+=1
		else:
			Feature=1
			if(Month==12):
				Month=1
				CurrentYear+=1
			else:
				Month+=1
			cols.append('Y_'+str(CurrentYear)+'_M_'+str(Month)+'_F_'+str(Feature))
			Feature+=1
	if (isTesting==False):
		cols.append('Target')
	if(isTesting and AnsFile!=None):
		df=DataFrame(Ansinfo, columns=cols)
		df.to_csv(Loc+AnsFile+'.csv',index=False)

	df=DataFrame(newinfo, columns=cols)
	df.to_csv(Loc+OuputFile+'.csv',index=False)




def changeMultiToUnivariate(inputfile,OuputFile):
	data = Loc+inputfile+ '.csv'
	dataframe = read_csv(data, names=None)
	dataframeCols = dataframe.columns.values
	info=dataframe.values
	cols =[]
	for i in range(info.shape[1]):
		if ("Target" in dataframeCols[i] or "F_1" in dataframeCols[i]):
			cols.append(i)
	newinfo = info[:,cols]
	newCols = dataframeCols[cols]
	df=DataFrame(newinfo, columns=newCols)
	df.to_csv(Loc+OuputFile+'.csv',index=False)






# divideTStoModelData('electricityDSTestingTS' , 'processedTestingEDS' , 4,isTesting=True)
# divideTStoModelData('electricityDSTrainingTS' , 'processedTrainingEDS' , 4)
# # getTimeSeries('electricityDSTesting' , 'electricityDSTestingTS'  )
