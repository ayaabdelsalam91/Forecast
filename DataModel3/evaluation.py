import torch
from elecLSTM import modelLSTMElec
import sys
sys.path.append('../Forecasting_DataModel2/Code/')
import Bias
import numpy as np

def getInputs():
    kmm = Bias.kMeanBias_("TrainingDS", 3, 104, 16, hasMonth=True)[2]
    print("opening file")
    myfile = open('../Forecasting_DataModel2/Data/TestingDS.csv','r')
    lines = []
    i = 0
    print("testing set opened")
    for line in myfile:
        if i == 0:
            i += 1
            continue
        toadd = list(map(float,line.split(",")))
        lines.append(toadd)
    i = 0
    inputs= []
    while i < len(lines):

        inputElec,f1,f2,f3 = [],[],[],[]
        arr = lines[i]


        j = 0
        end = 0
        while j < len(arr):
            vals = np.array([arr[j]]+[arr[j+13]]+[arr[j+14]]+[arr[j+15]])
            currCenter = Bias.getOneCenter_(vals,kmm)
            inputElec.append(arr[j+0:j+13]+[currCenter[0][0]])
            f1.append([arr[j+13]]+arr[j+1:j+13]+[currCenter[0][1]])
            f2.append([arr[j+14]]+arr[j+1:j+13]+[currCenter[0][2]])
            f3.append([arr[j+15]]+arr[j+1:j+13]+[currCenter[0][3]])

            if arr[j] == 0 and end == 0:
                end = j
            j+=16


        inputs.append([inputElec, f1, f2, f3])

        i += 1
    return inputs

torch.nn.Module.dump_patches = True
model = torch.load('./FinalModels/modelElecNoBias.pth')
arr = getInputs()