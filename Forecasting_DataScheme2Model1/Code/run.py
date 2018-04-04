import numpy as np
from prepareElectrictyData import UpdateTempList ,  addTemp , divideDataToTrainingAndTesting, getTimeSeries , divideTStoModelData , changeMultiToUnivariate , RemoveFeatures
from Model1  import LSTM as Model1_LSTM
from Model1 import getLSTMData as Model1_getLSTMData
from Bias import kMeanBias , averageBias ,getK  ,  kMeanBias_ , averageBias_
from helper import getNormializedData , getSubsetOfData 
import time
from Baseline import LSTM as Baseline_LSTM




# getNormializedData('TrainingElectrictyDSReOrdered' ,'TrainingElectrictyDSReOrderedNormalized',3)
# maxV_ , minV_ = getNormializedData('electricityDSTraining' ,'electricityDSTrainingNormalized',3)
# getNormializedData('electricityDSTesting' ,'electricityDSTestingNormalized',3,flag=False , maxV =  maxV_ , minV =  minV_)
# getTimeSeries('electricityDSTestingNormalized' , 'electricityDSTestingTS'  )
# getTimeSeries('electricityDSTrainingNormalized' , 'electricityDSTrainingTS'  ) 

# divideTStoModelData('electricityDSTestingTS' , 'TestingElectrictyDS' , 4,isTesting=True , AnsFile = 'TestingElectrictyDSAns' )
# divideTStoModelData('electricityDSTrainingTS' , 'TrainingElectrictyDS' , 4)

# changeMultiToUnivariate('TrainingElectrictyDS' , 'TrainingElectrictyDS_Uni')
# changeMultiToUnivariate('TestingElectrictyDS' , 'TestingElectrictyDS_Uni')
# changeMultiToUnivariate('TestingElectrictyDSAns', 'TestingElectrictyDSAns_Uni')
# changeMultiToUnivariate('TestingElectrictyDS_small' , 'TestingElectrictyDS_small_Uni')
# changeMultiToUnivariate('TestingElectrictyDSAns_small', 'TestingElectrictyDSAns_small_Uni')

# rows = getSubsetOfData ('TestingElectrictyDS' ,'TestingElectrictyDS_small')
# rows = getSubsetOfData ('TestingElectrictyDSAns' ,'TestingElectrictyDSAns_small' , rows=rows)


# RemoveFeatures('TrainingElectrictyDS', 'TrainingElectrictyDS_NoTemp' , Features = ['F_2'])
# RemoveFeatures('TestingElectrictyDS', 'TestingElectrictyDS_NoTemp' , Features = ['F_2'])
# RemoveFeatures('TestingElectrictyDSAns', 'TestingElectrictyDSAns_NoTemp' , Features = ['F_2'])
# RemoveFeatures('TestingElectrictyDS_small', 'TestingElectrictyDS_small_NoTemp' , Features = ['F_2'])
# RemoveFeatures('TestingElectrictyDSAns_small', 'TestingElectrictyDSAns_small_NoTemp' , Features = ['F_2'])


#########################
# TrainData , TrainOutput , Trainseq_length ,  TestData, Testseq_length , RID = Model1_getLSTMData ('TrainingElectrictyDS_NoTemp' , 'TestingElectrictyDS_small_NoTemp', 3, timeSteps = 129 ,
#  	isTargetReplication = False , hasID=False)


# K ,  data= getK('TrainingElectrictyDSReOrderedNormalized_NoTemp')
# KMeanBias ,  KmeanBaseline ,  model = kMeanBias('TrainingElectrictyDSReOrderedNormalized_NoTemp',K=K)
# AverageBias ,  averageBaseline = averageBias('TrainingElectrictyDSReOrderedNormalized_NoTemp')


# t0 = time.time()


# Model1_LSTM('EDSTestOuputM1_200_small_newKmean_NoTemp' , TrainData , TrainOutput , Trainseq_length ,  TestData, Testseq_length , 
#  	learning_rate = 0.001,n_neurons=64, n_layers = 2 , alpha=0.25,n_epochs=200,  trainKeepProb = 1.0, isBaseline =KmeanBaseline , Bias1=AverageBias , Bias2=KMeanBias ,DS = 2 ,  testing=True , KmeanModel = model)

# t1 = time.time()

# total = t1-t0
# print (total)
####################################


TrainData , TrainOutput , Trainseq_length ,  TestData, Testseq_length , RID = Model1_getLSTMData ('TrainingElectrictyDS' , 'TestingElectrictyDS_small', 4, timeSteps = 129 ,
 	isTargetReplication = False , hasID=False)


K ,  data= getK('TrainingElectrictyDSReOrderedNormalized')
# KMeanBias ,  KmeanBaseline ,  model = kMeanBias_('TrainingElectrictyDS',2,129,4)
# AverageBias ,  averageBaseline = averageBias_('TrainingElectrictyDS' , 4)


# t0 = time.time()


# Model1_LSTM('EDSTestOuputM1_200_small_newK2_' , TrainData , TrainOutput , Trainseq_length ,  TestData, Testseq_length , 
#  	learning_rate = 0.001,n_neurons=64, n_layers = 2 , alpha=0.25,n_epochs=200,  trainKeepProb = 1.0, isBaseline =KmeanBaseline , Bias1=AverageBias , Bias2=KMeanBias ,DS = 2 ,  testing=True , KmeanModel = model)

# t1 = time.time()

# total = t1-t0
# print (total)
# # ####################################
# t0 = time.time()


#TrainData , TrainOutput , Trainseq_length ,  TestData, Testseq_length , RID = Model1_getLSTMData ('TrainingElectrictyDS_Uni' , 'TestingElectrictyDS_small_Uni', 1, timeSteps = 129 ,
#  	isTargetReplication = False , hasID=False)




# Baseline_LSTM('EDSTestOuputBaseline_500_small_' , TrainData , TrainOutput , Trainseq_length ,  TestData, Testseq_length , 
#  	learning_rate = 0.001,n_neurons=64, n_layers = 2 , alpha=0.25,n_epochs=500,  trainKeepProb = 1.0, DS = 2 ,  testing=True)


# t1 = time.time()

# total = t1-t0
# print (total)