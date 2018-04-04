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

FeaturesCount = 1
TrainData , TrainOutput , Trainseq_length ,  TestData, Testseq_length , RID = Model1_getLSTMData ('TrainingDS_Uni' , 'TestingDS_Uni', FeaturesCount, timeSteps = 104, isTargetReplication = False , hasID=False )

print(TrainData.shape)


print(Testseq_length)

Baseline_LSTM('BL_500_UNI' , TrainData , TrainOutput , Trainseq_length ,  TestData, Testseq_length , 
 	learning_rate = 0.0001,n_neurons=64, n_layers = 2 , alpha=0.25,n_epochs=500,  trainKeepProb = 1.0, DS = 2 ,  testing=True ,hasMonth=False)


t1 = time.time()

total = t1-t0
print (total)


os.system('say "your program has finished"')
