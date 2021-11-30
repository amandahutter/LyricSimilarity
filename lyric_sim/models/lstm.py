from numpy.core.numeric import outer
import torch 
import torch.nn as nn 

class LSTM(nn.Module):
    def __init__(self, input_size, emb_size=20, hidden_size=10, dropout=0.90, num_fc = 1, combo_unit = 'MULT'):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.emb_size = emb_size 
        self.dropout = dropout 
        self.num_fc = num_fc
        self.combo_unit = combo_unit

        self.embedding = nn.Embedding(input_size, emb_size)
        self.dropout = nn.Dropout(p = dropout)        
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first = True)

        # Variable size of Fully Connected Layer, dependent on Combination Unit 
        if combo_unit == 'ADD':
            self.multiplier = 2
        elif combo_unit == 'SUB':
            self.multiplier = 1
        elif combo_unit == 'MULT':
            self.multiplier = 1
        else: 
            self.multiplier = 4 

        self.fc_first =  nn.Linear(self.multiplier * hidden_size, self.multiplier * hidden_size)
        self.relu = nn.ReLU()
        self.fc_final = nn.Linear(self.multiplier  * hidden_size, 2)
        self.softmax = nn.Softmax(dim = 1)
        # Should there be CrossEntropyLoss here, instead of softmax? 

        self.h_f_1 = None 
        self.h_f_2 = None 
        self.h_f = None 
    
    def network(self, song):

        N, T, H = song.shape
        h_n = torch.zeros(N, self.hidden_size)
        c_n = torch.zeros(N, self.hidden_size)
        for t in range(0, T):
            word =  song[:, t, :]
            embedded = self.embedding(word)
            dropped = self.dropout(embedded)
            outputs, h_n, c_n = self.lstm(dropped, h_n, c_n)
            h_n = self.dropout(h_n)
        return h_n 

    def forward(self, input1, input2):
        # input1 and input 2 are the songs 
        
        self.h_f_1 = self.network(input1)
        self.h_f_2 = self.network(input2)

        
        # Combine these hidden layers together 

        # if ADD, shape is 2*dh 
        if self.combo_unit == 'ADD': 
            self.h_f = torch.cat((self.h_f_1, self.h_f_2), dim = 1)
        # if SUB, shape is dh
        elif self.combo_unit == 'SUB':
            self.h_f = torch.sub(self.h_f_1, self.h_f_2)
        # if MULT, shape is dh 
        elif self.combo_unit == 'MULT': 
            self.h_f = torch.mul(self.h_f_1, self.h_f_2)
        # if All, shape is 4*dh 
        else: 
            add = torch.cat((self.h_f_1, self.h_f_2), dim = 1)
            sub = torch.sub(self.h_f_1, self.h_f_2)
            mult = torch.mul(self.h_f_1, self.h_f_2)
            self.h_f = torch.cat((add,sub,mult), dim = 1)


        # Fully Connected Layers 
        if self.num_fc == 1:
            self.h_f = self.fc_final(self.h_f)
        else: # f_c == 2 
            self.h_f = self.fc_first(self.h_f)
            self.h_f = self.dropout(self.h_f)
            self.h_f = self.relu(self.h_f)
            self.h_f = self.fc_final(self.h_f)
    
        output = self.softmax(self.h_f)

        return output 