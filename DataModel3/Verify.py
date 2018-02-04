import torch
import GenerateInputs
from torch import nn
from torch import autograd

useGPU = False
class modelLSTMVent(nn.Module):

    def __init__(self,input_size, hidden_dim, num_layers):
        super(modelLSTMVent, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers)
        self.lstm2 = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers)
        self.lstm3 = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers)
        self.lstm4 = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers)
        self.lstm5 = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers)
        self.lstm6 = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers)
        self.lstm7 = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers)
        self.lstm8 = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers)
        self.lstm9 = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers)
        self.lstm10 = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers)
        self.lstm11 = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers)
        self.lstmMain = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers)

        self.featureLayer1 = nn.Linear(hidden_dim, 1)
        self.featureLayer2 = nn.Linear(hidden_dim, 1)
        self.featureLayer3 = nn.Linear(hidden_dim, 1)
        self.featureLayer4 = nn.Linear(hidden_dim, 1)
        self.featureLayer5 = nn.Linear(hidden_dim, 1)
        self.featureLayer6 = nn.Linear(hidden_dim, 1)
        self.featureLayer7 = nn.Linear(hidden_dim, 1)
        self.featureLayer8 = nn.Linear(hidden_dim, 1)
        self.featureLayer9 = nn.Linear(hidden_dim, 1)
        self.featureLayer10 = nn.Linear(hidden_dim, 1)
        self.featureLayer11 = nn.Linear(hidden_dim, 1)

        self.targetLayer = nn.Linear(12 * 32, 1)


    # def init_all_hiddens(self):
    #     self.lstm1.hidden = self.init_hidden()
    #     self.lstm2.hidden = self.init_hidden()
    #     self.lstm3.hidden = self.init_hidden()
    #     self.lstm4.hidden = self.init_hidden()
    #     self.lstm5.hidden = self.init_hidden()
    #     self.lstm6.hidden = self.init_hidden()
    #     self.lstm7.hidden = self.init_hidden()
    #     self.lstm8.hidden = self.init_hidden()
    #     self.lstm9.hidden = self.init_hidden()
    #     self.lstm10.hidden = self.init_hidden()
    #     self.lstm11.hidden = self.init_hidden()


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
            input[i] = autograd.Variable(torch.FloatTensor(input[i])).view(20,1,-1)
            i += 1

        lstm1hidden = self.init_hidden()
        lstm2hidden = self.init_hidden()
        lstm3hidden = self.init_hidden()
        lstm4hidden = self.init_hidden()
        lstm5hidden = self.init_hidden()
        lstm6hidden = self.init_hidden()
        lstm7hidden = self.init_hidden()
        lstm8hidden = self.init_hidden()
        lstm9hidden = self.init_hidden()
        lstm10hidden = self.init_hidden()
        lstm11hidden = self.init_hidden()
        lstmMainhidden = self.init_hidden()

        for i in range(20):
            if float(input[0][i].data) == 0.0:
                break
            inputMain = input[0][i].view(1,1,-1)
            inputf1 = input[1][i].view(1, 1, -1)
            inputf2 = input[2][i].view(1, 1, -1)
            inputf3 = input[3][i].view(1, 1, -1)
            inputf4 = input[4][i].view(1, 1, -1)
            inputf5 = input[5][i].view(1, 1, -1)
            inputf6 = input[6][i].view(1, 1, -1)
            inputf7 = input[7][i].view(1, 1, -1)
            inputf8 = input[8][i].view(1, 1, -1)
            inputf9 = input[9][i].view(1, 1, -1)
            inputf10 = input[10][i].view(1, 1, -1)
            inputf11 = input[11][i].view(1, 1, -1)

            lstmMain_out, lstmMainhidden = self.lstmMain(inputMain,lstmMainhidden)
            lstm1_out, lstm1hidden = self.lstm1(inputf1, lstm1hidden)
            lstm2_out, lstm2hidden = self.lstm2(inputf2, lstm2hidden)
            lstm3_out, lstm3hidden = self.lstm3(inputf3, lstm3hidden)
            lstm4_out, lstm4hidden = self.lstm4(inputf4,  lstm4hidden)
            lstm5_out, lstm5hidden = self.lstm5(inputf5, lstm5hidden)
            lstm6_out, lstm6hidden = self.lstm6(inputf6, lstm6hidden)
            lstm7_out, lstm7hidden = self.lstm7(inputf7, lstm7hidden)
            lstm8_out, lstm8hidden = self.lstm8(inputf8, lstm8hidden)
            lstm9_out, lstm9hidden = self.lstm9(inputf9, lstm9hidden)
            lstm10_out, lstm10hidden = self.lstm10(inputf10, lstm10hidden)
            lstm11_out, lstm11hidden = self.lstm11(inputf11, lstm11hidden)


        targetIn = torch.cat((  lstmMain_out,lstm1_out,lstm2_out,lstm3_out,lstm4_out,lstm5_out,lstm6_out,
                                lstm7_out,lstm8_out,lstm9_out,lstm10_out,lstm11_out)).view(-1,self.hidden_dim*12)

        targetOut = self.targetLayer(targetIn)

        f1Out = self.featureLayer1(lstm1_out)
        f2Out = self.featureLayer2(lstm2_out)
        f3Out = self.featureLayer2(lstm3_out)
        f4Out = self.featureLayer2(lstm4_out)
        f5Out = self.featureLayer2(lstm5_out)
        f6Out = self.featureLayer2(lstm6_out)
        f7Out = self.featureLayer2(lstm7_out)
        f8Out = self.featureLayer2(lstm8_out)
        f9Out = self.featureLayer2(lstm9_out)
        f10Out = self.featureLayer2(lstm10_out)
        f11Out = self.featureLayer2(lstm11_out)

        featuresOut = [f1Out,f2Out,f3Out,f4Out,f5Out,f6Out,f7Out,f8Out,f9Out,f10Out,f11Out]

        return targetOut,featuresOut

    def zero_all_lstm_grads(self):
        self.lstm1.zero_grad()
        self.lstm2.zero_grad()
        self.lstm3.zero_grad()
        self.lstm4.zero_grad()
        self.lstm5.zero_grad()
        self.lstm6.zero_grad()
        self.lstm7.zero_grad()
        self.lstm8.zero_grad()
        self.lstm9.zero_grad()
        self.lstm10.zero_grad()
        self.lstm11.zero_grad()


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
        term2 = (1-alpha)*(y-targets[0])
        return alpha*term1 + (1-alpha)*term2

inputs,targets = GenerateInputs.getInputs()

model = torch.load('C:/PythonProj/LSTM-LongTermForecasting/model.pth')

input1 = inputs[300]
y,features = model.forward(inputs[300])
print(y,features)

