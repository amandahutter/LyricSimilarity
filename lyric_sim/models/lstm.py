from numpy.core.numeric import outer
import torch 
import torch.nn as nn 
from enum import Enum

class CombinationType(Enum):
  ADD = 1
  SUB = 2
  MULT = 3
  ALL = 4

class LSTM(nn.Module):


    def __init__(self, input_size, emb_size=20, hidden_size=10, dropout=0.90, num_fc=1, combo_unit=CombinationType.MULT):

        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.emb_size = emb_size 
        self.dropout = dropout 
        self.num_fc = num_fc
        self.combo_unit = combo_unit

        self.embedding = nn.Embedding(input_size, emb_size)
        self.dropout = nn.Dropout(p = dropout)        
        #self.lstm = nn.LSTM(emb_size, hidden_size, batch_first = True)
        #self.lstm = nn.LSTM(emb_size, hidden_size, batch_first = True, dropout = dropout)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first = True, num_layers = 2, dropout = dropout)

        # Variable size of Fully Connected Layer, dependent on Combination Unit 
        if combo_unit == CombinationType.ADD:
            self.multiplier = 2
        elif combo_unit == CombinationType.SUB:
            self.multiplier = 1
        elif combo_unit == CombinationType.MULT:
            self.multiplier = 1
        elif combo_unit == CombinationType.ALL:
            self.multiplier = 4
        else: 
            print("Invalid combination argument. Defaulting to all.")
            self.multiplier = 4 
        self.hidden_mult = self.multiplier * self.hidden_size

        self.fc_first =  nn.Linear(self.hidden_mult, self.hidden_mult)
        self.relu = nn.ReLU()
        self.fc_final = nn.Linear(self.hidden_mult, 2)

        self.h_f_1 = None 
        self.h_f_2 = None 
        self.h_f = None 
    
    def network(self, song):

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        song = song.to(device)
        
        N, T = song.shape
        embedded = self.embedding(song)
        #dropped = self.dropout(embedded)
        #_, (h_n, c_n) = self.lstm(dropped)
        _, (h_n, c_n) = self.lstm(embedded)
        h_n = self.dropout(h_n)
        h_n = h_n[-1]
        h_n = h_n.unsqueeze(dim = 0)
                    
        return h_n 

    def forward(self, input1, input2):
        
        self.h_f_1 = self.network(input1)
        self.h_f_2 = self.network(input2)
        
        # Combine these hidden layers together 
        # if ADD, shape is 2*dh 
        if self.combo_unit == CombinationType.ADD:
            self.h_f = torch.cat((self.h_f_1, self.h_f_2), dim = 2)
        # if SUB, shape is dh
        elif self.combo_unit == CombinationType.SUB:
            self.h_f = torch.sub(self.h_f_1, self.h_f_2)
        # if MULT, shape is dh 
        elif self.combo_unit == CombinationType.MULT:
            self.h_f = torch.mul(self.h_f_1, self.h_f_2)
        # if All, shape is 4*dh 
        elif self.combo_unit == CombinationType.ALL:
            add = torch.cat((self.h_f_1, self.h_f_2), dim = 2)
            sub = torch.sub(self.h_f_1, self.h_f_2)
            mult = torch.mul(self.h_f_1, self.h_f_2)
            self.h_f = torch.cat((add,sub,mult), dim = 2)
        else: 
            print("Invalid combination argument. Defaulting to all.")
            add = torch.cat((self.h_f_1, self.h_f_2), dim = 2)
            sub = torch.sub(self.h_f_1, self.h_f_2)
            mult = torch.mul(self.h_f_1, self.h_f_2)
            self.h_f = torch.cat((add,sub,mult), dim = 2)


        # Fully Connected Layers 
        if self.num_fc == 1:
            self.h_f = self.fc_final(self.h_f)
        else: # f_c == 2 
            self.h_f = self.fc_first(self.h_f)
            self.h_f = self.dropout(self.h_f)
            self.h_f = self.relu(self.h_f)
            self.h_f = self.fc_final(self.h_f)
    
        #output = self.softmax(self.h_f)
        output = self.h_f

        return output 