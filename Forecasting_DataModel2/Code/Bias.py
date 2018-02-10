import sklearn.metrics as sm
import numpy as np
from pandas import read_csv, DataFrame
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import  silhouette_score

import time



Loc = '../Forecasting_DataModel2/Data/'



def averageBias(inputfile):
	valuesOfIntrest=[]
	isBaseLine =[] 
	data =Loc+inputfile+'.csv'

	dataframe = read_csv(data, names=None)
	info=dataframe.values
	
	for i in range(info.shape[1]):
		if(dataframe.columns.values[i]!="RID" and dataframe.columns.values[i]!="VISCODE" 
		 and dataframe.columns.values[i]!="Year" and dataframe.columns.values[i]!="State" and   dataframe.columns.values[i]!="Month" ):
			unq = np.unique(info[:,i])

			if(len(unq)>50):
				print('HERE!')
				
				valuesOfIntrest.append(np.mean(info[:,i]))
			else:
				valuesOfIntrest.append(stats.mode(info[:,i])[0].flatten()[0])
			if("bl" in  dataframe.columns.values[i] ):
				isBaseLine.append(1)
			else:
				isBaseLine.append(0)

	return valuesOfIntrest , isBaseLine 



def averageBias_(inputfile,numberOfFeatures):
	isBaseLine =[] 
	data =Loc+inputfile+'.csv'
	dataframe = read_csv(data, names=None)
	info=dataframe.values
	cols =  dataframe.columns.values
	lists=[]
	for k in range(1,numberOfFeatures+1):
		index=[]
		for j in range(info.shape[1]):
			if("F_"+str(k) in cols[j]):
				index.append(j)
		lists.append(index)

	avg = []
	for i in range(numberOfFeatures):
		matrix= info[:,lists[i]]
		matrix = matrix.reshape(matrix.shape[0]*matrix.shape[1])
		avg.append(np.nanmean(np.where(matrix!=0,matrix,np.nan)))
	for i in range(info.shape[1]):
		if(dataframe.columns.values[i]!="RID" and dataframe.columns.values[i]!="VISCODE" 
		 and dataframe.columns.values[i]!="Year" and dataframe.columns.values[i]!="State" and   dataframe.columns.values[i]!="Month" ):
			if("bl" in  dataframe.columns.values[i] ):
				isBaseLine.append(1)
			else:
				isBaseLine.append(0)
	return avg , isBaseLine

def kMeanBias(inputfile,K):

	isBaseLine =[] 
	data =Loc+inputfile+'.csv'
	dataframe = read_csv(data, names=None)
	info=dataframe.values
	for i in range(info.shape[1]):
		if(dataframe.columns.values[i]!="RID" and dataframe.columns.values[i]!="VISCODE" 
		 and dataframe.columns.values[i]!="Year" and dataframe.columns.values[i]!="State" and   dataframe.columns.values[i]!="Month" ):
			start=i
			break

	inputToKMeans = info[:,start:]
	print(inputToKMeans.shape)
	kmeans = KMeans(n_clusters=K).fit(inputToKMeans)

	for i in range(info.shape[1]):
		if(dataframe.columns.values[i]!="RID" and dataframe.columns.values[i]!="VISCODE" 
		 and dataframe.columns.values[i]!="Year" and dataframe.columns.values[i]!="State" and   dataframe.columns.values[i]!="Month" ):
			unq = np.unique(info[:,i])
			if("bl" in  dataframe.columns.values[i] ):
				isBaseLine.append(1)
			else:
				isBaseLine.append(0)
	return kmeans.cluster_centers_ , isBaseLine  , kmeans




def kMeanBias_(inputfile,K,n_steps,numberOfFeatures,FixedLength=False,hasMonth=False):
	data =Loc+inputfile+'.csv'
	dataframe = read_csv(data, names=None)
	info=dataframe.values
	info=info[:,:-1]
	
	cols = dataframe.columns.values
	if(not FixedLength):

		if(hasMonth):

			MonthList = []
			for i in range(2,14):
				MonthList.append("F_"+str(i))
			FeatureList=[]
			for i in range(len(cols)-1):
				Flag=0
				for j in range(len(MonthList)):
					if(MonthList[j] in cols[i]):
						Flag=1
						break
				if(Flag==0):
					FeatureList.append(i)
			numberOfFeatures=numberOfFeatures-12
			info = info[:,FeatureList]

		info=info.reshape(info.shape[0] ,n_steps,numberOfFeatures)
		count = 0
		for i in range (info.shape[0]):
			for j in range(n_steps):
				if(np.sum(info[i,j,:])!=0):
					count+=1
		trainSet = np.zeros((count,numberOfFeatures))
	else:
		if(hasMonth):
			MonthList = []
			for i in range(2,14):
				MonthList.append("F_"+str(i))
			FeatureList=[]
			for i in range(len(cols)-1):
				Flag=0
				for j in range(len(MonthList)):
					if(MonthList[j] in cols[i]):
						Flag=1
						break
				if(Flag==0):
					FeatureList.append(i)
			numberOfFeatures=numberOfFeatures-12
			info = info[:,FeatureList]
			trainSet = np.zeros((info.shape[0]*n_steps,numberOfFeatures))
			info=info.reshape(info.shape[0] ,n_steps,numberOfFeatures)

	index=0
	for i in range (info.shape[0]):
		for j in range(n_steps):
			if(np.sum(info[i,j,:])!=0):
				trainSet[index] = info[i,j,:]
				index+=1
	for i in range(trainSet.shape[1]):
		print(np.sum(trainSet[:,i]))
	kmeans = KMeans(n_clusters=K).fit(trainSet)

	isBaseLine =[] 
	for i in range(info.shape[1]):
		if(dataframe.columns.values[i]!="RID" and dataframe.columns.values[i]!="VISCODE" 
		 and dataframe.columns.values[i]!="Year" and dataframe.columns.values[i]!="State" and   dataframe.columns.values[i]!="Month" ):
			unq = np.unique(info[:,i])
			if("bl" in  dataframe.columns.values[i] ):
				isBaseLine.append(1)
			else:
				isBaseLine.append(0)


	return kmeans.cluster_centers_ , isBaseLine  , kmeans

def getCenter_(input , model):
	# print(input.shape)
	steps = input.shape[1]
	# print(steps , input.shape)
	# print (centers.shape , input.shape , numberOfCenters ,steps)
	output = np.zeros((steps))
	input = input.reshape((input.shape[1],input.shape[2]))
	for i in range(input.shape[0]):
		# print(i, input[i,:].shape)
		output[i] = model.predict(input[i,:].reshape(1, -1))[0]
	counts = np.bincount(output.astype(int))
	index =np.argmax(counts)
	center = model.cluster_centers_[index]
	# print ("new" , center ,  index)
	return center


def getOneCenter_(input , model):

	index = model.predict(input.reshape(1, -1))[0]
	center = model.cluster_centers_[index]
	return center ,  index 





def getCenter(input , centers):
	steps = input.shape[1]
	# print(steps , input.shape)
	numberOfCenters  =centers.shape[0]
	output = np.zeros((steps,numberOfCenters))
	# print (centers.shape , input.shape , numberOfCenters ,steps)
	input = input.reshape((input.shape[1],input.shape[2]))
	# t0 = time.time()
	for i in range (steps):
		 output[i]=np.linalg.norm(input[i,:]-centers,axis=1)
	if(steps>1):
		max = np.argmax(output,axis=1)
		index = stats.mode(max)[0][0]


	else:
		index =np.argmax(output)
	center = centers[index,:]
	# t1 = time.time()

	# total = t1-t0
	# print (total , steps , index , output.shape)
	print ("old" , center ,  index)
	return center



def getK(inputfile):
	data =Loc+inputfile+'.csv'
	dataframe = read_csv(data, names=None)
	info=dataframe.values
	for i in range(info.shape[1]):
		if(dataframe.columns.values[i]!="RID" and dataframe.columns.values[i]!="VISCODE" 
		 and dataframe.columns.values[i]!="Year" and dataframe.columns.values[i]!="State" and   dataframe.columns.values[i]!="Month" ):
			start=i
			break
	inputToKMeans = info[:,start:]
	range_n_clusters=[]
	for i in range(2,10):
		range_n_clusters.append(i)

	score = []
	for n_clusters in range_n_clusters:
		clusterer = KMeans(n_clusters=n_clusters, random_state=10)
		cluster_labels = clusterer.fit_predict(inputToKMeans)
		silhouette_avg = silhouette_score(inputToKMeans, cluster_labels)
		print("\item For n clusters =", n_clusters,
          "The average silhouette score is :", silhouette_avg)
		score.append(silhouette_avg)
	score = np.array(score)
	maxIdex = np.argmax(score)
	print (maxIdex, score[maxIdex] ,  range_n_clusters[maxIdex])
	return range_n_clusters[maxIdex] , info











