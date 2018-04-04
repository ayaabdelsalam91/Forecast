import numpy as np
from prepareElectrictyData import UpdateTempList ,  addTemp , divideDataToTrainingAndTesting, getTimeSeries , divideTStoModelData , changeMultiToUnivariate , RemoveFeatures
from Model1_Modified  import LSTM as Model1_LSTM
from Model1_Modified import getLSTMData as Model1_getLSTMData
from Bias import kMeanBias , averageBias ,getK  ,  kMeanBias_ , averageBias_
from helper import getNormializedData , getSubsetOfData 
import time
from Baseline import LSTM as Baseline_LSTM


KMeanBias ,  KmeanBaseline ,  Kmeanmodel = kMeanBias_('TrainingElectrictyDS',2,129,4)
AverageBias ,  averageBaseline = averageBias_('TrainingElectrictyDS' , 4)


TrainData , TrainOutput , Trainseq_length ,  TestData, Testseq_length , RID = Model1_getLSTMData ('TrainingElectrictyDS' , 'TestingElectrictyDS_small', 4, timeSteps = 129 ,
  	isTargetReplication = False , hasID=False ,Bias=AverageBias )


print(TrainData.shape , TestData.shape)



t0 = time.time()


Model1_LSTM('EDSTestOuputM1_400_small_ConcatFeatures_AVG' , TrainData , TrainOutput , Trainseq_length ,  TestData, Testseq_length , 
 	learning_rate = 0.001,n_neurons=64, n_layers = 2 , alpha=0.25,n_epochs=1,  trainKeepProb = 1.0 ,DS = 2 ,  testing=True , Bias = AverageBias)

t1 = time.time()

total = t1-t0
print (total)
