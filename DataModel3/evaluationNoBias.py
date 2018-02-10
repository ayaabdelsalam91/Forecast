import torch
from elecLSTM import modelLSTMElec
import sys
sys.path.append('../Forecasting_DataModel2/Code/')
import Bias
import numpy as np

def getInputs():
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
            inputElec.append(arr[j+0:j+13])
            f1.append([arr[j+13]]+arr[j+1:j+13])
            f2.append([arr[j+14]]+arr[j+1:j+13])
            f3.append([arr[j+15]]+arr[j+1:j+13])

            if arr[j] == 0 and end == 0:
                end = j
            j+=16


        inputs.append([inputElec, f1, f2, f3])

        i += 1
    return inputs

model = torch.load('./FinalModels/modelElecNoBias.pth')
arr = getInputs()

months = [  [0,0,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0,0,0,0,0],
            [0,0,0,1,0,0,0,0,0,0,0,0],
            [0,0,0,0,1,0,0,0,0,0,0,0],
            [0,0,0,0,0,1,0,0,0,0,0,0],
            [0,0,0,0,0,0,1,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,0,0,0,0],
            [0,0,0,0,0,0,0,0,1,0,0,0],
            [0,0,0,0,0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,0,0,0,0,1,0]   ]

matElec = np.zeros((51,24))
matF1 = np.zeros((51,24))
matF2 = np.zeros((51,24))
matF3 = np.zeros((51,24))

for i in range(len(arr)):
    print(i)
    input = arr[i]
    target,features = model.forward(input)
    matElec[i][0] = target.data[0][0]
    matF1[i][0] = features[0].data[0]
    matF2[i][0] = features[1].data[0]
    matF3[i][0] = features[2].data[0]
    for j in range(1,24):

        toAppend = [[matElec[i][j - 1]] + months[(j-1) % len(months)]]
        toAppend = torch.autograd.Variable(torch.FloatTensor(toAppend)).view(1, 1, -1)
        input[0] = torch.cat((input[0], toAppend))

        toAppend2 = [[matF1[i][j - 1]] + months[(j-1) % len(months)]]
        toAppend2 = torch.autograd.Variable(torch.FloatTensor(toAppend2)).view(1, 1, -1)
        input[1] = torch.cat((input[1], toAppend2))

        toAppend3 = [[matF2[i][j - 1]] + months[(j-1) % len(months)]]
        toAppend3 = torch.autograd.Variable(torch.FloatTensor(toAppend3)).view(1, 1, -1)
        input[2] = torch.cat((input[2], toAppend3))

        toAppend4 = [[matF3[i][j - 1]] + months[(j-1) % len(months)]]
        toAppend4 = torch.autograd.Variable(torch.FloatTensor(toAppend4)).view(1, 1, -1)
        input[3] = torch.cat((input[3], toAppend4))

        target,features = model.forward(input)
        matElec[i][j] = target.data[0][0]
        matF1[i][j] = features[0].data[0]
        matF2[i][j] = features[1].data[0]
        matF3[i][j] = features[2].data[0]

ansMat = np.loadtxt(open("../Forecasting_DataModel2/Data/TestingAnsDS.csv", "rb"), delimiter=",", skiprows=1)
residualMatrix = np.abs(ansMat-matElec)
np.savetxt("noBiasErrors.csv",X=residualMatrix,delimiter=",",newline="\n")