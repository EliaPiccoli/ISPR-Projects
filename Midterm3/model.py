# https://github.com/zutotonno/char-rnn.pytorch

import torch
import torch.nn as nn
import random
import math
from torch.autograd import Variable

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model="gru", n_layers=1,
     dropout=0, gpu=True, batch_size=32, chunk_len=30, learning_rate=0.001, optimizer="adam"):
        super(CharRNN, self).__init__()
        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.gpu = gpu
        self.batch_size = batch_size
        self.chunk_len = chunk_len
        self.optimizer = optimizer

        self.encoder = nn.Embedding(input_size, hidden_size)

        if self.model == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        elif self.model == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=dropout)
        elif self.model == "rnn":
            self.rnn = nn.RNN(hidden_size, hidden_size, n_layers, dropout=dropout)

        self.decoder = nn.Linear(hidden_size, output_size)
        
        if self.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        elif self.optimizer == "rms":
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        if self.gpu:
            self.cuda()

    def forward(self, input, hidden):
        batch_size = input.size(0)
        encoded = self.encoder(input)
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden

    def init_hidden(self, batch_size):
        if self.model == "lstm":
             return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
        return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))

    
    def train(self,inp, target, validation, teacher_forcing=False, teacher_ratio=0.5):
        self.zero_grad()
        loss = 0
        accuracy = 0
        hidden = self.init_hidden(self.batch_size)
        predicted = None
        if self.cuda:
            if self.model == "gru" or self.model == "rnn":
                hidden = hidden.cuda()
            else:
                hidden = (hidden[0].cuda(), hidden[1].cuda())
        next_inp = inp[:, 0]
        for c in range(self.chunk_len):
            output, hidden = self(next_inp, hidden)
            _, predicted = torch.max(output.view(self.batch_size, -1).data, 1)
            accuracy += (predicted == target[:, c]).sum().item()
            loss += self.criterion(output.view(self.batch_size, -1), target[:, c])       
            if c+1 < self.chunk_len:
                if teacher_forcing:
                    r = math.floor(random.random())
                    if r > teacher_ratio:
                        # print("\nUsing prediction: ", c, teacher_ratio, r)
                        next_inp = predicted
                    else:
                        next_inp = inp[:, c+1]
                else:
                    next_inp = inp[:, c+1]
        if not validation:
            loss.backward()
            self.optimizer.step()
        currentLoss = loss.item()/(self.chunk_len*predicted.size(0))
        currentAcc = accuracy/(self.chunk_len*predicted.size(0))
        return currentLoss, currentAcc