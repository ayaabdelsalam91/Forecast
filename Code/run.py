import numpy as np
from prepareElectrictyData import UpdateTempList ,  addTemp , divideDataToTrainingAndTesting, getTimeSeries , divideTStoModelData , changeMultiToUnivariate
from Model1  import LSTM as Model1_LSTM
from Model1 import getLSTMData as Model1_getLSTMData
from Bias import kMeanBias , averageBias ,getK
from helper import getNormializedData
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
changeMultiToUnivariate('TestingElectrictyDSAns', 'TestingElectrictyDSAns_Uni')

# TrainData , TrainOutput , Trainseq_length ,  TestData, Testseq_length , RID = Model1_getLSTMData ('TrainingElectrictyDS_Uni' , 'TestingElectrictyDS_Uni', 1, timeSteps = 129 ,
#  	isTargetReplication = False , hasID=False)


TrainData , TrainOutput , Trainseq_length ,  TestData, Testseq_length , RID = Model1_getLSTMData ('TrainingElectrictyDS' , 'TestingElectrictyDS', 4, timeSteps = 129 ,
 	isTargetReplication = False , hasID=False)

# print (TrainData.shape ,TrainOutput.shape ,  Trainseq_length.shape , TestData.shape, Testseq_length.shape )
# # for i in range(10):
# # 	print (i , TrainData [i,:Trainseq_length[i]-1] ,  TrainOutput[i, :Trainseq_length[i]-1])


# K= getK('TrainingElectrictyDSReOrderedNormalized')
KMeanBias ,  KmeanBaseline = kMeanBias('TrainingElectrictyDSReOrderedNormalized',K=2)
AverageBias ,  averageBaseline = averageBias('TrainingElectrictyDSReOrderedNormalized')


# t0 = time.time()


Model1_LSTM('EDSTestOuputM1_200_' , TrainData , TrainOutput , Trainseq_length ,  TestData, Testseq_length , 
 	learning_rate = 0.001,n_neurons=64, n_layers = 2 , alpha=0.25,n_epochs=200,  trainKeepProb = 1.0, isBaseline =KmeanBaseline , Bias1=AverageBias , Bias2=KMeanBias ,DS = 2 ,  testing=True)

# Baseline_LSTM('EDSTestOuputBaseline_' , TrainData , TrainOutput , Trainseq_length ,  TestData, Testseq_length , 
#  	learning_rate = 0.001,n_neurons=64, n_layers = 2 , alpha=0.25,n_epochs=1000,  trainKeepProb = 1.0, DS = 2 ,  testing=True)


# t1 = time.time()

# total = t1-t0
# print (total)