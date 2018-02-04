import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import GenerateInputsElec
import inspect


useGPU = False
SAVEPATH = './modelElec.pth'

print(torch.cuda.is_available())
class modelLSTMElec(nn.Module):

    def __init__(self,input_size, hidden_dim, num_layers):
        super(modelLSTMElec, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers)
        self.lstm2 = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers)
        self.lstm3 = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers)
        self.lstmMain = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers)

        self.featureLayer1 = nn.Linear(hidden_dim, 1)
        self.featureLayer2 = nn.Linear(hidden_dim, 1)
        self.featureLayer3 = nn.Linear(hidden_dim, 1)

        self.targetLayer = nn.Linear(4 * 32, 1)

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if useGPU:
            return (autograd.Variable(torch.zeros(self.num_layers, 1, self.hidden_dim).cuda()),
                    autograd.Variable(torch.zeros(self.num_layers, 1, self.hidden_dim).cuda()))
        else:
            return (autograd.Variable(torch.zeros(self.num_layers, 1, self.hidden_dim)),
                    autograd.Variable(torch.zeros(self.num_layers, 1, self.hidden_dim)))

    def forward(self, input):
        i = 0


        while i < len(input):
            if not isinstance(input[i],autograd.Variable):
                input[i] = autograd.Variable(torch.FloatTensor(input[i])).view(129,1,-1)
            i += 1

        lstm1hidden = self.init_hidden()
        lstm2hidden = self.init_hidden()
        lstm3hidden = self.init_hidden()
        lstmMainhidden = self.init_hidden()

        for i in range(129):
            if float(input[0][i].data) == 0.0:
                break
            if useGPU:
                inputMain = input[0][i].view(1, 1, -1).cuda()
                inputf1 = input[1][i].view(1, 1, -1).cuda()
                inputf2 = input[2][i].view(1, 1, -1).cuda()
                inputf3 = input[3][i].view(1, 1, -1).cuda()
            else:
                inputMain = input[0][i].view(1,1,-1)
                inputf1 = input[1][i].view(1, 1, -1)
                inputf2 = input[2][i].view(1, 1, -1)
                inputf3 = input[3][i].view(1, 1, -1)

            lstmMain_out, lstmMainhidden = self.lstmMain(inputMain,lstmMainhidden)
            lstm1_out, lstm1hidden = self.lstm1(inputf1, lstm1hidden)
            lstm2_out, lstm2hidden = self.lstm2(inputf2, lstm2hidden)
            lstm3_out, lstm3hidden = self.lstm3(inputf3, lstm3hidden)


        if useGPU:
            targetIn = torch.cat((lstmMain_out,lstm1_out,lstm2_out,lstm3_out)).view(-1,self.hidden_dim * 4).cuda()
        else:
            targetIn = torch.cat((lstmMain_out, lstm1_out, lstm2_out, lstm3_out)).view(-1,self.hidden_dim * 4)

        targetOut = self.targetLayer(targetIn)

        f1Out = self.featureLayer1(lstm1_out)
        f2Out = self.featureLayer2(lstm2_out)
        f3Out = self.featureLayer2(lstm3_out)

        featuresOut = [f1Out,f2Out,f3Out]

        return targetOut,featuresOut

    def zero_all_lstm_grads(self):
        self.lstm1.zero_grad()
        self.lstm2.zero_grad()
        self.lstm3.zero_grad()
        self.lstmMain.zero_grad()


    def custom_loss(self,x,y,targets,alpha):
        term1 = None
        i = 1
        while i < len(targets):
            if term1 is None:
                term1 = (x[i-1] - (targets[i])) ** 2
            else:
                term1 += (x[i-1] - (targets[i])) ** 2
            i += 1
        term1 = alpha*term1
        term2 = (y-targets[0])**2
        return alpha*term1 + (1-alpha)*term2


if useGPU:
    model = modelLSTMElec(1, 32, 2).cuda()
else:
    model = modelLSTMElec(1, 32, 2)
inputs,targets = GenerateInputsElec.getInputs()
optimizer = optim.SGD(model.parameters(), lr=0.1)



def train():
    first=True
    for epoch in range(200):
        i= 0
        epochLoss = 0
        while i < len(inputs):
            model.zero_grad()
            model.zero_all_lstm_grads()

            if i % 10 == 0:
                print(i)


            input1 = inputs[i]
            currentTargets = targets[i]
            if first:
                j = 0
                while j < len(currentTargets):
                    if useGPU:
                        currentTargets[j] = autograd.Variable(torch.FloatTensor(currentTargets[j])).view(-1).cuda()
                    else:
                        currentTargets[j] = autograd.Variable(torch.FloatTensor(currentTargets[j])).view(-1)
                    j += 1

            yhat,xhats = model.forward(input1)

            loss = model.custom_loss(xhats,yhat,currentTargets,0.5)
            loss.backward()
            epochLoss += float(loss.data[0])
            optimizer.step()
            i = i+1
        first=False
        torch.save(model, SAVEPATH)
        print("epoch #"+str(epoch)+" loss = "+str(epochLoss/len(inputs)))

train()