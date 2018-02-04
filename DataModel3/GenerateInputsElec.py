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
    means,stds = getNums()
    print("opening file")
    myfile = open('./Data/TrainingElectrictyDS.csv','r')
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
        zeroIdx = getZeroIndex(arr)


        j = 0
        while j < len(arr):
            if j >= zeroIdx:
                inputElec.append(arr[j])
                f1.append(arr[j + 1])
                f2.append(arr[j + 2])
                f3.append(arr[j + 3])
            else:
                inputElec.append((arr[j] - means[0]) / stds[0])
                f1.append((arr[j + 1] - means[1]) / stds[1])
                f2.append((arr[j + 2] - means[2]) / stds[2])
                f3.append((arr[j + 3] - means[3]) / stds[3])
            j += 4

        if zeroIdx != 516:
            nextArr = lines[i+1][:-1]
            t1.append((nextArr[zeroIdx + 1] - means[1]) / stds[1])
            t2.append((nextArr[zeroIdx + 2] - means[2]) / stds[2])
            t3.append((nextArr[zeroIdx + 3] - means[3]) / stds[3])
        else:
            t1.append((arr[zeroIdx - 3] - means[3]) / stds[3])
            t2.append((arr[zeroIdx - 2] - means[2]) / stds[2])
            t3.append((arr[zeroIdx - 1] - means[1]) / stds[1])

        inputs.append([inputElec, f1, f2, f3])
        toadd = float(arr2[-1])
        toadd = [(toadd - means[0]) / stds[0]]
        targets.append([toadd, t1, t2, t3])

        i += 1
    return inputs,targets



#vals,targets = getInputs()
#print(retval[0],retval[1])