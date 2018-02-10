def getZeroIndex(array):
    i = 0
    while i < len(array):
        if array[i] == 0:
            return i
        i += 1
    return i

def getInputs():
    print("opening file")
    myfile = open('./LSTMDataVentriclesForecastLeaderBoardTrainParsed.csv','r')
    lines = []
    i = 0
    print("got here1.vent")
    for line in myfile:
        if i == 0:
            i += 1
            continue
        toadd = list(map(float,line.split(",")))[1:]
        lines.append(toadd)
    i = 0
    inputs,targets = [],[]
    while i < len(lines):

        inputVent,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11 = [],[],[],[],[],[],[],[],[],[],[],[]
        t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11 = [],[],[],[],[],[],[],[],[],[],[]
        arr2 = lines[i]
        arr = arr2[:-1]
        zeroIdx = getZeroIndex(arr)


        j = 0
        while j < len(arr):
            inputVent.append(arr[j])
            f1.append(arr[j + 1])
            f2.append(arr[j + 2])
            f3.append(arr[j + 3])
            f4.append(arr[j + 4])
            f5.append(arr[j + 5])
            f6.append(arr[j + 6])
            f7.append(arr[j + 7])
            f8.append(arr[j + 8])
            f9.append(arr[j + 9])
            f10.append(arr[j + 10])
            f11.append(arr[j + 11])
            j += 12

        if zeroIdx != 240:
            nextArr = lines[i+1][:-1]
            t1.append(nextArr[zeroIdx + 1])
            t2.append(nextArr[zeroIdx + 2])
            t3.append(nextArr[zeroIdx + 3])
            t4.append(nextArr[zeroIdx + 4])
            t5.append(nextArr[zeroIdx + 5])
            t6.append(nextArr[zeroIdx + 6])
            t7.append(nextArr[zeroIdx + 7])
            t8.append(nextArr[zeroIdx + 8])
            t9.append(nextArr[zeroIdx + 9])
            t10.append(nextArr[zeroIdx + 10])
            t11.append(nextArr[zeroIdx + 11])
        else:
            t1.append(arr[zeroIdx - 11])
            t2.append(arr[zeroIdx - 10])
            t3.append(arr[zeroIdx - 9])
            t4.append(arr[zeroIdx - 8])
            t5.append(arr[zeroIdx - 7])
            t6.append(arr[zeroIdx - 6])
            t7.append(arr[zeroIdx - 5])
            t8.append(arr[zeroIdx - 4])
            t9.append(arr[zeroIdx - 3])
            t10.append(arr[zeroIdx - 2])
            t11.append(arr[zeroIdx - 1])

        inputs.append([inputVent,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11])
        targets.append([[float(arr2[-1])],t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11])

        i += 1
    print("got here2")
    return inputs,targets