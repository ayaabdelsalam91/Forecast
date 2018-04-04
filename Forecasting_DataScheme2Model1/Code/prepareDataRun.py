from prepareElectrictyData import UpdateTempList ,  addTemp , divideDataToTrainingAndTesting, getTimeSeries , divideTStoModelData , changeMultiToUnivariate , RemoveFeatures , getNormializedData , getDiff


##################### Graph Plot #################################


# getNormializedData('electricityDataset' ,'electricityDatasetNormalized',3)
# getTimeSeries('electricityDatasetNormalized' , 'electricityDSTS')
# changeMultiToUnivariate('electricityDSTS','electricityDSTS_uni', KeyWord = "Megawatthours" )


################################ Normal ####################################
# getDiff('electricityDataset' , 'electricityDataset_Diff')
# FeaturesCount = 16
# getNormializedData('electricityDataset_Diff' ,'electricityDatasetNormalized',3,AddMonthFlag =True ,  AddStateFlag=False)
# divideDataToTrainingAndTesting('electricityDatasetNormalized','electricityDS')
# getTimeSeries('electricityDSTraining','electricityDSTrainingTS' ,startFeature ="M_1" )
# getTimeSeries('electricityDSTesting','electricityDSTestingTS',isTesting=True ,  startFeature ="M_1" )
# divideTStoModelData('electricityDSTestingTS' , 'TestingDS' , FeaturesCount*2,isTesting=True , AnsFile = 'TestingAnsDS' , NumberOfFeatures=FeaturesCount)
# divideTStoModelData('electricityDSTrainingTS' , 'TrainingDS' , FeaturesCount , NumberOfFeatures=FeaturesCount)


# #################### Baseline ###############################
# FeaturesCount = 13
# Features = ['Temp'	 , 'Count'	, 'price']
# RemoveFeatures('electricityDataset_Diff','electricityDataset_Month',Features)
# getNormializedData('electricityDataset_Uni' ,'electricityDatasetNormalized_Month',3 ,AddMonthFlag =True )
# divideDataToTrainingAndTesting('electricityDatasetNormalized_Month','electricityDS_Month')
# getTimeSeries('electricityDS_MonthTraining','electricityDSTrainingTS_Month' ,startFeature ="M_1" )
# getTimeSeries('electricityDS_MonthTesting','electricityDSTestingTS_Month',isTesting=True ,  startFeature ="M_1" )
# divideTStoModelData('electricityDSTestingTS_Month' , 'TestingDS_Month' , FeaturesCount*2,isTesting=True , AnsFile = 'TestingAnsDS_Month' , NumberOfFeatures=FeaturesCount)
# divideTStoModelData('electricityDSTrainingTS_Month' , 'TrainingDS_Month' , FeaturesCount , NumberOfFeatures=FeaturesCount)


# # #################### -Temp ###############################
# FeaturesCount = 15
# Features = ['Temp']
# RemoveFeatures('electricityDataset_Diff','electricityDataset_NotTemp',Features)
# getNormializedData('electricityDataset_NotTemp' ,'electricityDatasetNormalized_NotTemp',3 ,AddMonthFlag =True )
# divideDataToTrainingAndTesting('electricityDatasetNormalized_NotTemp','electricityDS_NotTemp')
# getTimeSeries('electricityDS_NotTempTraining','electricityDSTrainingTS_NotTemp' ,startFeature ="M_1" )
# getTimeSeries('electricityDS_NotTempTesting','electricityDSTestingTS_NotTemp',isTesting=True ,  startFeature ="M_1" )
# divideTStoModelData('electricityDSTestingTS_NotTemp' , 'TestingDS_NotTemp' , FeaturesCount*2,isTesting=True , AnsFile = 'TestingAnsDS_NotTemp' , NumberOfFeatures=FeaturesCount)
# divideTStoModelData('electricityDSTrainingTS_NotTemp' , 'TrainingDS_NotTemp' , FeaturesCount , NumberOfFeatures=FeaturesCount)



# #################### -TempAndPrice ###############################
FeaturesCount = 14
Features = ['Temp' , 'price']
RemoveFeatures('electricityDataset_Diff','electricityDataset_NotTempprice',Features)
getNormializedData('electricityDataset_NotTempprice' ,'electricityDatasetNormalized_NotTempprice',3 ,AddMonthFlag =True )
divideDataToTrainingAndTesting('electricityDatasetNormalized_NotTempprice','electricityDS_NotTempprice')
getTimeSeries('electricityDS_NotTemppriceTraining','electricityDSTrainingTS_NotTempprice' ,startFeature ="M_1" )
getTimeSeries('electricityDS_NotTemppriceTesting','electricityDSTestingTS_NotTempprice',isTesting=True ,  startFeature ="M_1" )
divideTStoModelData('electricityDSTestingTS_NotTempprice' , 'TestingDS_NotTempprice' , FeaturesCount*2,isTesting=True , AnsFile = 'TestingAnsDS_NotTempprice' , NumberOfFeatures=FeaturesCount)
divideTStoModelData('electricityDSTrainingTS_NotTempprice' , 'TrainingDS_NotTempprice' , FeaturesCount , NumberOfFeatures=FeaturesCount)