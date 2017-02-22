import torch
import torch.nn as nn
from torch.autograd import Variable

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnncell_type, ntoken, ninp, nhid, nlayers):
        super(RNNModel, self).__init__()
        self.encoder = nn.Embedding(ntoken, ninp)
        #self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, bias=False)
        self.rnncell = getattr(nn, rnncell_type)(ninp, nhid, bias=False)
        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights()

        self.rnn_type = rnncell_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.encoder(input)
        #output, hidden = self.rnn(emb, hidden)

        #hidden: 2*20*200
        hidden = hidden.squeeze(0)
        
        seq_len = input.size(0)
        batch_size = input.size(1)
        output_dim = hidden.size(1)

        output = Variable(torch.Tensor(seq_len, batch_size, output_dim))
        #print(hidden.size())
        c = Variable(hidden.data)
        for i in range(seq_len):
            # print("loop...")
            # print(emb[i].size())
            # print(hidden.size())
            # print(c.size())
            
            hidden, c = self.rnncell(emb[i], (hidden, c))
            #print(output.data.size())
            #print(hidden.data.size())
            output.data[i] = hidden.data


        #output = torch.FloatTensor(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
