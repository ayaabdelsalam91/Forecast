import numpy as np
from Bias import kMeanBias , averageBias ,getK 
import time
import numpy as np
from pandas import read_csv, DataFrame
from sklearn.cluster import KMeans

KMeanBias ,  KmeanBaseline ,  model = kMeanBias('TrainingElectrictyDSReOrderedNormalized',K=2)
AverageBias ,  averageBaseline = averageBias('TrainingElectrictyDSReOrderedNormalized')

print(KMeanBias)
# Loc = '../Data/'
# data =Loc+'TrainingElectrictyDS'+'.csv'
# dataframe = read_csv(data, names=None)
# info=dataframe.values
# cols =  dataframe.columns.values
# sum = 0
# count=0
# list=[]
# for k in range(1,5):
# 	sum = 0
# 	count=0
# 	for j in range(info.shape[1]):
# 		if("F_"+str(k) in cols[j]):
# 			sum+=np.sum(info[:,j])
# 			for i in range(info.shape[0]):
# 				if(info[i,j]!=0):
# 					count+=1
# 	list.append(sum/count )
# print(list)


# lists=[]
# for k in range(1,5):
# 	index=[]
# 	for j in range(info.shape[1]):
# 		if("F_"+str(k) in cols[j]):
# 			index.append(j)
# 	lists.append(index)

# avg = []
# for i in range(4):
# 	matrix= info[:,lists[i]]

# 	matrix = matrix.reshape(matrix.shape[0]*matrix.shape[1])
# 	avg.append(np.nanmean(np.where(matrix!=0,matrix,np.nan)))
# print(avg)



Loc = '../Data/'
data =Loc+'TrainingElectrictyDS'+'.csv'
dataframe = read_csv(data, names=None)
info=dataframe.values
print(info.shape)
info=info[:,:-1]
info=info.reshape(info.shape[0] ,129,4)
count = 0
all = 0
for i in range (info.shape[0]):
	for j in range(129):
		if(np.sum(info[i,j,:])!=0):
			count+=1
		all+=1
print(count , all)
trainSet = np.zeros((count,4))
index=0
for i in range (info.shape[0]):
	for j in range(129):
		if(np.sum(info[i,j,:])!=0):
			trainSet[index] = info[i,j,:]
			index+=1
kmeans = KMeans(n_clusters=2).fit(trainSet)
print(kmeans.cluster_centers_)

# centers=[]
# for i in range(info.shape[0]):
# 	center=[]
# 	for j in range (info.shape[1]):
# 		center.append(model.predict(info[i,j,:].reshape(1, 4) )[0])
# 	centers.append(center)

# for center in centers:
# 	print(center)

# import matplotlib.pyplot as plt

# with plt.style.context('fivethirtyeight'):
#     for center in centers:
#         plt.plot(range(len(center)), center)

# plt.show()





# Loc = '../Data/'
# data =Loc+'TrainingElectrictyDS'+'.csv'
# dataframe = read_csv(data, names=None)
# info=dataframe.values
# print(info.shape)
# info=info.reshape(info.shape[0] ,129,4)
# # centers=[]
# # for i in range(info.shape[0]):
# # 	center=[]
# # 	for j in range (info.shape[1]):
# # 		center.append(model.predict(info[i,j,:].reshape(1, 4) )[0])
# # 	centers.append(center)

# # for center in centers:
# # 	print(center)

# # import matplotlib.pyplot as plt

# # with plt.style.context('fivethirtyeight'):
# #     for center in centers:
# #         plt.plot(range(len(center)), center)

# # plt.show()

# plt.imshow(centers)
# plt.show()
# for i in range(info.shape[1]):
# 	if(dataframe.columns.values[i]!="RID" and dataframe.columns.values[i]!="VISCODE" 
# 	 and dataframe.columns.values[i]!="Year" and dataframe.columns.values[i]!="State" and   dataframe.columns.values[i]!="Month" ):
# 		start=i
# 		break

# inputToKMeans = info[:,start:]
# for i in range(inputToKMeans.shape[0]):
# 	print(model.predict(inputToKMeans[i].reshape(1, -1)) , model.predict(inputToKMeans[i].reshape(1, 4)) )