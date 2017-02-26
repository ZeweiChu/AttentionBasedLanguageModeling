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

        #output linear transformation
        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights()

        self.rnn_type = rnncell_type
        self.nhid = nhid
        self.nlayers = nlayers

        # self.linear1 = nn.Linear()
        self.bilinear_term = nn.Parameter(torch.Tensor(nhid, nhid).random_())

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.encoder(input)
        #output, hidden = self.rnn(emb, hidden)

        #hidden: 2*20*200
        h, c = hidden
        h.data = h.data.squeeze(0)
        c.data = c.data.squeeze(0)

        
        seq_len = input.size(0)
        batch_size = input.size(1)
        output_dim = h.size(1)


        output = []
        for i in range(seq_len):
            # print("loop...")
            # print(emb[i].size())
            # print(hidden.size())
            # print(c.size())
            
            h, c = self.rnncell(emb[i], (h, c))
            o = Variable(torch.zeros(h.data.size()))
            # hiddens: time * batch * nhid
            if i == 0:
                hiddens = h.unsqueeze(0)
            else:
                #print(hiddens.size())
                #print(h.size())
                hiddens = torch.cat([hiddens, h.unsqueeze(0)])

            h2 = h.unsqueeze(0).expand_as(hiddens)
            att = torch.sum(hiddens * h2, 2).expand_as(hiddens)
            o = att * hiddens
            o = torch.sum(o, 0)

            output.append(o) #hidden.data

        output = torch.cat(output)
        #print(output.size())

        #output = torch.FloatTensor(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        
        if self.rnn_type == 'LSTMCell':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
