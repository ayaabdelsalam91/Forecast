import sys
sys.path.append('../Forecasting_DataModel2/Code/')
import Bias

model = Bias.kMeanBias_("TrainingDS",3,104,16,hasMonth=True)
print("done")