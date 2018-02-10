import numpy as np

def getZeroIndex(array):
    i = 0
    while i < len(array):
        if array[i] == 0:
            return i
        i += 1
    return i

def getNums():
    print("opening file")
    myfile = open('../Data/TrainingDS.csv', 'r')
    lines = []
    i = 0
    for line in myfile:
        if i == 0:
            i += 1
            continue
        toadd = list(map(float, line.split(",")))
        lines.append(toadd)

    arr1,arr2,arr3,arr4 = [],[],[],[]

    for line in lines:
        i = 0
        while i < len(line)-1:
            if line[i] == 0:
                break
            arr1.append(line[i])
            arr2.append(line[i+1])
            arr3.append(line[i+2])
            arr4.append(line[i+3])
            i += 4

    return [np.mean(arr1),np.mean(arr2),np.mean(arr3),np.mean(arr4)]\
            ,[np.std(arr1),np.std(arr2),np.std(arr3),np.std(arr4)]

def getInputs():
    print("opening file")
    myfile = open('../Forecasting_DataModel2/Data/TrainingDS.csv','r')
    lines = []
    i = 0
    print("got here.elec1")
    for line in myfile:
        if i == 0:
            i += 1
            continue
        toadd = list(map(float,line.split(",")))
        lines.append(toadd)
    i = 0
    inputs,targets = [],[]
    while i < len(lines):

        inputElec,f1,f2,f3 = [],[],[],[]
        t1,t2,t3 = [],[],[]
        arr2 = lines[i]
        arr = arr2[:-1]


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

        if arr[-1] == 0 and i != len(lines)-1:
            t1.append(lines[i+1][end+13])
            t2.append(lines[i+1][end+14])
            t3.append(lines[i+1][end+15])
        else:
            t1.append(lines[i][-2])
            t2.append(lines[i][-3])
            t3.append(lines[i][-4])


        inputs.append([inputElec, f1, f2, f3])
        toadd = float(arr2[-1])
        toadd = [(toadd)]
        targets.append([toadd, t1, t2, t3])

        i += 1
    return inputs,targets



#vals,targets = getInputs()
#print(retval[0],retval[1])