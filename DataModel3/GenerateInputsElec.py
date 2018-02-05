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
    myfile = open('../Data/TrainingDS.csv','r')
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
        while j < len(arr):
            inputElec.append(arr[j])

            #add one hot vector for month to input of each feature
            f1.append(arr[j + 1])
            f1.append(arr[j + 2])
            f1.append(arr[j + 3])
            f1.append(arr[j + 4])
            f1.append(arr[j + 5])
            f1.append(arr[j + 6])
            f1.append(arr[j + 7])
            f1.append(arr[j + 8])
            f1.append(arr[j + 9])
            f1.append(arr[j + 10])
            f1.append(arr[j + 11])
            f1.append(arr[j + 12])
            #add the feature itself
            f1.append(arr[j + 13])

            f2.append(arr[j + 1])
            f2.append(arr[j + 2])
            f2.append(arr[j + 3])
            f2.append(arr[j + 4])
            f2.append(arr[j + 5])
            f2.append(arr[j + 6])
            f2.append(arr[j + 7])
            f2.append(arr[j + 8])
            f2.append(arr[j + 9])
            f2.append(arr[j + 10])
            f2.append(arr[j + 11])
            f2.append(arr[j + 12])
            #
            f2.append(arr[j + 14])

            f3.append(arr[j + 1])
            f3.append(arr[j + 2])
            f3.append(arr[j + 3])
            f3.append(arr[j + 4])
            f3.append(arr[j + 5])
            f3.append(arr[j + 6])
            f3.append(arr[j + 7])
            f3.append(arr[j + 8])
            f3.append(arr[j + 9])
            f3.append(arr[j + 10])
            f3.append(arr[j + 11])
            f3.append(arr[j + 12])
            #
            f3.append(arr[j + 15])
            j += 16

        if i != len(lines)-1:
            t1.append(lines[i+1][j+13])
            t2.append(lines[i+1][j+14])
            t3.append(lines[i+1][j+15])
        else:
            t1.append(lines[i][j + 13])
            t2.append(lines[i][j + 14])
            t3.append(lines[i][j + 15])

        inputs.append([inputElec, f1, f2, f3])
        toadd = float(arr2[-1])
        toadd = [(toadd)]
        targets.append([toadd, t1, t2, t3])

        i += 1
    return inputs,targets



vals,targets = getInputs()
#print(retval[0],retval[1])