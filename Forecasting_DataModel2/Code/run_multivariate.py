import numpy as np
from prepareElectrictyData import UpdateTempList ,  addTemp , divideDataToTrainingAndTesting, getTimeSeries , divideTStoModelData , changeMultiToUnivariate , RemoveFeatures
from Model1_Modified  import LSTM as Model1_LSTM
from Model1_Modified import getLSTMData as Model1_getLSTMData
from Bias import kMeanBias , averageBias ,getK  ,  kMeanBias_ , averageBias_
from helper import getNormializedData , getSubsetOfData 
import time
from Baseline import LSTM as Baseline_LSTM
import os


t0 = time.time()

FeaturesCount = 16


# KMeanBias ,  KmeanBaseline ,  Kmeanmodel = kMeanBias_('TrainingDS',2,104,FeaturesCount,hasMonth=True)
getK('electricityDSTrainingGetK')
# AverageBias ,  averageBaseline = averageBias_('TrainingElectrictyDS' , 4)


# print(KMeanBias)
# TrainData , TrainOutput , Trainseq_length ,  TestData, Testseq_length , RID = Model1_getLSTMData ('TrainingDS' , 'TestingDS', FeaturesCount, timeSteps = 104 ,
#   	isTargetReplication = False , hasID=False ,model=Kmeanmodel,hasMonth=True )


# print(TrainData.shape ,TrainOutput.shape , Trainseq_length.shape  ,  TestData.shape  ,Testseq_length.shape)


# print(KMeanBias)


# # # (OutputFile  , X_, Y_,Trainseq_length, TestData , Testseq_length ,
# # # 	learning_rate ,n_neurons, n_layers , alpha,n_epochs,  trainKeepProb = 1.0 ,isBaseline =None , Bias=None ,RID=None  , DS = 1 ,  testing=True,KmeanModel=None,hasMonth=False)
# Model1_LSTM('B2_500' , TrainData , TrainOutput  ,  Trainseq_length , TestData , Testseq_length, 
#  	learning_rate = 0.0001,n_neurons=64, n_layers = 2 , alpha=0.8,n_epochs=500,  trainKeepProb = 1.0 , isBaseline =KmeanBaseline, DS = 2 , testing=True , Bias = None ,KmeanModel=Kmeanmodel  , hasMonth=True)


t1 = time.time()

total = t1-t0
print (total)


# os.system('say "your program has finished"')