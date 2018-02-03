from prepareElectrictyData import UpdateTempList ,  addTemp , divideDataToTrainingAndTesting, getTimeSeries , divideTStoModelData , changeMultiToUnivariate , RemoveFeatures , getNormializedData , getDiff


##################### Graph Plot #################################


# getNormializedData('electricityDataset' ,'electricityDatasetNormalized',3)
# getTimeSeries('electricityDatasetNormalized' , 'electricityDSTS')
# changeMultiToUnivariate('electricityDSTS','electricityDSTS_uni', KeyWord = "Megawatthours" )


################################ Normal ####################################
#getDiff('electricityDataset' , 'electricityDataset_Diff')
FeaturesCount = 16
# getNormializedData('electricityDataset_Diff' ,'electricityDatasetNormalized',3,AddMonthFlag =True ,  AddStateFlag=False)
# divideDataToTrainingAndTesting('electricityDatasetNormalized','electricityDS')
# getTimeSeries('electricityDSTraining','electricityDSTrainingTS' ,startFeature ="M_1" )
# getTimeSeries('electricityDSTesting','electricityDSTestingTS',isTesting=True ,  startFeature ="M_1" )
divideTStoModelData('electricityDSTestingTS' , 'TestingDS' , FeaturesCount*2,isTesting=True , AnsFile = 'TestingAnsDS' , NumberOfFeatures=FeaturesCount)
#divideTStoModelData('electricityDSTrainingTS' , 'TrainingDS' , FeaturesCount , NumberOfFeatures=FeaturesCount)


# #################### Baseline ###############################
# FeaturesCount = 12+1+51
# Features = ['Temp'	 , 'Count'	, 'price']
# RemoveFeatures('electricityDataset','electricityDataset_Uni',Features)
# getNormializedData('electricityDataset_Uni' ,'electricityDatasetNormalized_Uni',3,AddMonthFlag =True)

# divideDataToTrainingAndTesting('electricityDatasetNormalized_Uni','electricityDS_uniAndMonthAndState')
# getTimeSeries('electricityDS_uniAndMonthAndStateTraining','electricityDSTrainingTS_UniAndMonthAndState' ,startFeature ="M_1" )
# getTimeSeries('electricityDS_uniAndMonthAndStateTesting','electricityDSTestingTS_UniAndMonthAndState',isTesting=True ,  startFeature ="M_1" )
# divideTStoModelData('electricityDSTestingTS_UniAndMonthAndState' , 'TestingDS_UniAndMonthAndState' , FeaturesCount,isTesting=True , AnsFile = 'TestingAnsDS_UniAndMonth' , NumberOfFeatures=FeaturesCount)
# divideTStoModelData('electricityDSTrainingTS_UniAndMonthAndState' , 'TrainingDS_UniAndMonthAndState' , FeaturesCount , NumberOfFeatures=FeaturesCount)