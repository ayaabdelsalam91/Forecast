import sklearn.metrics as sm
import numpy as np
from pandas import read_csv, DataFrame
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import  silhouette_score

import time



Loc = '../Data/'



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
				
				valuesOfIntrest.append(np.mean(info[:,i]))
			else:
				valuesOfIntrest.append(stats.mode(info[:,i])[0].flatten()[0])
			if("bl" in  dataframe.columns.values[i] ):
				isBaseLine.append(1)
			else:
				isBaseLine.append(0)

	return valuesOfIntrest , isBaseLine 


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
	kmeans = KMeans(n_clusters=K).fit(inputToKMeans)

	for i in range(info.shape[1]):
		if(dataframe.columns.values[i]!="RID" and dataframe.columns.values[i]!="VISCODE" 
		 and dataframe.columns.values[i]!="Year" and dataframe.columns.values[i]!="State" and   dataframe.columns.values[i]!="Month" ):
			unq = np.unique(info[:,i])
			if("bl" in  dataframe.columns.values[i] ):
				isBaseLine.append(1)
			else:
				isBaseLine.append(0)
	return kmeans.cluster_centers_ , isBaseLine 

def getCenter(input , centers):
	

	steps = input.shape[1]
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
	# print (center ,  index)
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
	for i in range(2,20):
		range_n_clusters.append(i)

	score = []
	for n_clusters in range_n_clusters:
		clusterer = KMeans(n_clusters=n_clusters, random_state=10)
		cluster_labels = clusterer.fit_predict(inputToKMeans)
		silhouette_avg = silhouette_score(inputToKMeans, cluster_labels)
		print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
		score.append(silhouette_avg)
	score = np.array(score)
	maxIdex = np.argmax(score)
	print (maxIdex, score[maxIdex] ,  range_n_clusters[maxIdex])
	return range_n_clusters[maxIdex]











